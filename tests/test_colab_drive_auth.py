"""Offline tests for the Drive auth and mount helpers embedded in the Colab notebook."""
import ast
import json
from pathlib import Path

import unittest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "NeuroLens_Complete_Training_Pipeline.ipynb"


def _notebook_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = ast.parse("".join(notebook["cells"][3]["source"]))
    names = {"_drive_api_for_valid_user", "_resolve_drive_user", "_ensure_drivefs_mount"}
    nodes = [node for node in setup.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    return scope


class UserCredential:
    def __init__(self, valid=True, expired=False, refresh_token=None):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_count = 0

    def refresh(self, request):
        self.refresh_count += 1
        self.valid = True


class ServiceAccountCredential:
    valid = True


class FakeAboutRequest:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0

    def execute(self):
        self.calls += 1
        if self.failure:
            raise self.failure
        return {"user": {"emailAddress": "researcher@example.test"}}


class FakeDriveAPI:
    def __init__(self, about_request=None):
        self.request = about_request or FakeAboutRequest()

    def about(self):
        return self

    def get(self, **kwargs):
        return self.request


class FakeAPIBuilder:
    def __init__(self, api=None):
        self.api = api or FakeDriveAPI()
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.api


class DriveAuthTests(unittest.TestCase):
    def test_existing_mount_is_reused_without_calling_mount(self):
            helpers = _notebook_helpers()
            mount_calls = []
            mounted, error = helpers["_ensure_drivefs_mount"]("/content/drive", lambda _: True, mount_calls.append)
            assert mounted and error is None
            assert mount_calls == []


    def test_unmounted_drive_attempts_mount_once_and_confirms_mount(self):
            helpers = _notebook_helpers()
            checks = iter([False, True])
            mount_calls = []
            mounted, error = helpers["_ensure_drivefs_mount"]("/content/drive", lambda _: next(checks), mount_calls.append)
            assert mounted and error is None
            assert mount_calls == ["/content/drive"]


    def test_valid_user_credential_is_reused_without_oauth(self):
            helpers = _notebook_helpers()
            credential = UserCredential()
            api_builder = FakeAPIBuilder()
            auth_calls = []
            returned, api = helpers["_resolve_drive_user"](
                lambda: credential, lambda: auth_calls.append(True), UserCredential, object, api_builder
            )
            assert returned is credential and api is api_builder.api
            assert auth_calls == []
            assert api_builder.api.request.calls == 1  # about.get validated reuse


    def test_service_account_is_rejected_and_interactive_oauth_is_used(self):
            helpers = _notebook_helpers()
            service = ServiceAccountCredential()
            user = UserCredential()
            candidates = iter([service, user])
            auth_calls = []
            returned, api = helpers["_resolve_drive_user"](
                lambda: next(candidates), lambda: auth_calls.append(True), UserCredential, object, FakeAPIBuilder()
            )
            assert returned is user and api is not None
            assert auth_calls == [True]


    def test_about_get_failure_triggers_oauth_and_revalidates_user(self):
            helpers = _notebook_helpers()
            invalid_api = FakeAPIBuilder(FakeDriveAPI(FakeAboutRequest(RuntimeError("permission denied"))))
            valid_api = FakeAPIBuilder()
            user = UserCredential()
            candidates = iter([user, user])
            api_sequence = iter([invalid_api.api, valid_api.api])
            auth_calls = []
            returned, api = helpers["_resolve_drive_user"](
                lambda: next(candidates), lambda: auth_calls.append(True), UserCredential, object,
                lambda *args, **kwargs: next(api_sequence)
            )
            assert returned is user and api is valid_api.api
            assert auth_calls == [True]


    def test_authentication_failure_raises_actionable_error(self):
            helpers = _notebook_helpers()
            with self.assertRaisesRegex(RuntimeError, "Drive API authentication failed before dataset loading/training"):
                helpers["_resolve_drive_user"](
                    lambda: ServiceAccountCredential(),
                    lambda: (_ for _ in ()).throw(RuntimeError("OAuth canceled")),
                    UserCredential, object, FakeAPIBuilder()
                )

    def test_mount_exception_is_reported_as_drivefs_failure(self):
            helpers = _notebook_helpers()
            def broken_mount(_):
                raise OSError("DriveFS unavailable")
            mounted, error = helpers["_ensure_drivefs_mount"]("/content/drive", lambda _: False, broken_mount)
            assert not mounted
            assert isinstance(error, OSError)
            assert "DriveFS unavailable" in str(error)

if __name__ == "__main__":
    unittest.main()
