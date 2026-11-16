"""Lightweight synthetic tests for the optional research extensions."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neurolens.models.baselines import ResUNet3D, UNet3D
from neurolens.models.multitask import NeuroLens
from neurolens.losses.multitask_loss import MultiTaskLoss
from neurolens.evaluation.explainability import (GradCAM, activation_overlap,
    neurolens_2d_classification_cam, neurolens_2d_embedding_cam,
    neurolens_3d_embedding_cam, neurolens_3d_segmentation_cam)
from neurolens.evaluation.meta_learning import fit_classification_stack, select_segmentation_weights
from neurolens.training.experiment_registry import (
    load_prediction_cache, save_prediction_cache, update_experiment_registry,
    load_multimodel_probability_cache, save_multimodel_probability_cache,
)
from neurolens.training.checkpoint import load_checkpoint, save_checkpoint
from neurolens.training.segmentation import segmentation_objective


@pytest.mark.parametrize("model_cls", [ResUNet3D, UNet3D])
def test_3d_baseline_output_shape(model_cls):
    model = model_cls(in_channels=4, classes=4, base_channels=2).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 4, 32, 32, 32))["segmentation_logits"]
    assert output.shape == (1, 4, 32, 32, 32)
    assert torch.isfinite(output).all()




def test_segmentation_baseline_objective_backward():
    logits=torch.randn(1,4,8,8,8,requires_grad=True)
    target=torch.randint(0,4,(1,8,8,8))
    loss,parts=segmentation_objective(logits,target)
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in parts.values())
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

def test_neurolens_synthetic_forward_multitask_loss_and_gradcam():
    model = NeuroLens(in_channels=4, base_channels=2, projection_dim=8, classes=4, classification_classes=2)
    image2d = torch.randn(2, 4, 32, 32)
    image3d = torch.randn(2, 4, 32, 32, 32)
    output = model(image2d, image3d)
    assert output["classification_logits"].shape == (2, 2)
    assert output["embedding2d"].shape == (2, 8)
    assert output["embedding3d"].shape == (2, 8)
    assert output["segmentation_logits"].shape == (2, 4, 32, 32, 32)
    target = {"class_label":torch.tensor([0,1]), "segmentation":torch.randint(0,4,(2,32,32,32))}
    loss, parts = MultiTaskLoss(temperature=.1)(output,target)
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in parts.values())
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)
    model.eval()
    assert neurolens_2d_classification_cam(model)(image2d[:1],1).shape == (1,1,32,32)
    assert neurolens_2d_embedding_cam(model)(image2d[:1],0).shape == (1,1,32,32)
    assert neurolens_3d_embedding_cam(model)(image3d[:1],0).shape == (1,1,32,32,32)
    assert neurolens_3d_segmentation_cam(model)(image3d[:1],1).shape == (1,1,32,32,32)

def test_gradcam_2d_shape_and_finite():
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(),
                                 torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2))
    cam = GradCAM(model, model[1], lambda m, x: m(x))
    result = cam(torch.randn(2, 1, 16, 16), 1)
    assert result.shape == (2, 1, 16, 16)
    assert torch.isfinite(result).all()
    assert model.training  # original mode is restored after the explanation pass


def test_gradcam_3d_segmentation_target_shape_and_finite():
    class TinySeg(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.features = torch.nn.Conv3d(1, 3, 3, padding=1); self.head = torch.nn.Conv3d(3, 2, 1)
        def forward(self, x): return {"segmentation_logits": self.head(torch.relu(self.features(x)))}
    model = TinySeg().eval()
    cam = GradCAM(model, model.features, lambda m, x: m(x), logits_key="segmentation_logits")
    output = cam(torch.randn(1, 1, 8, 8, 8), 1, torch.ones(1, 8, 8, 8))
    assert output.shape == (1, 1, 8, 8, 8) and torch.isfinite(output).all()


def test_gradcam_embedding_coordinate_target():
    class TinyEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.features=torch.nn.Conv2d(1,4,3,padding=1); self.projection=torch.nn.Linear(4,3)
        def forward(self,x): return self.projection(torch.relu(self.features(x)).mean((2,3)))
    model=TinyEmbedding()
    cam=GradCAM(model,model.features,lambda m,x:{"embedding":m(x)},logits_key="embedding")
    result=cam(torch.randn(1,1,12,12),2)
    assert result.shape==(1,1,12,12) and torch.isfinite(result).all()


def test_activation_overlap_reports_patientwise_values():
    cam = torch.zeros(2, 1, 4, 4); cam[:, :, :2] = 1
    mask = torch.zeros(2, 4, 4, dtype=torch.long); mask[:, :2] = 1
    result = activation_overlap(cam, mask)
    assert result["dice"] == pytest.approx([1.0, 1.0])


def test_meta_stack_uses_oof_train_inputs_and_separate_validation():
    rng = np.random.default_rng(4)
    labels = np.asarray([0, 1] * 30)
    base = np.stack([labels * .8 + .1, 1 - labels * .8 - .1], axis=1)
    oof = np.stack([base, base], axis=1)
    val_y = np.asarray([0, 1] * 10)
    val_base = np.stack([val_y * .8 + .1, 1 - val_y * .8 - .1], axis=1)
    val = np.stack([val_base, val_base], axis=1)
    fitted = fit_classification_stack(oof, labels, val, val_y,
                                      candidates=("best_individual", "logistic_regression", "mean_probability", "weighted_probability"))
    assert fitted.selected_name in fitted.validation_scores
    assert "best_individual" in fitted.validation_scores
    assert fitted.predict_proba(val).shape == (20, 2)
    import inspect
    assert "test_labels" not in inspect.signature(fit_classification_stack).parameters


def test_segmentation_fusion_selects_on_validation_only():
    truth = np.zeros((2, 8, 8, 8), dtype=np.uint8); truth[:, 2:5, 2:5, 2:5] = 1
    good = np.eye(4, dtype=np.float32)[truth].transpose(0, 4, 1, 2, 3)
    bad = np.full_like(good, .25)
    selected = select_segmentation_weights({"good": good, "bad": bad}, truth)
    assert selected["selection_split"] == "validation"
    assert selected["selected"] in selected["candidates"]


def test_registry_and_prediction_cache_roundtrip(tmp_path):
    registry_path = tmp_path / "registry.json"
    update_experiment_registry(registry_path, "resunet-seed42", {"status": "done", "epoch": 3})
    update_experiment_registry(registry_path, "swin-seed42", {"status": "pending"})
    assert len(__import__("json").loads(registry_path.read_text())["experiments"]) == 2
    cache = tmp_path / "predictions.npz"
    ids = ["case-1", "case-2"]; probs = np.full((2, 3, 4, 2), .5, dtype=np.float32)
    save_prediction_cache(cache, ids, probs, {"architecture": "resunet", "split": "validation"})
    loaded = load_prediction_cache(cache, ids, {"architecture": "resunet", "split": "validation"})
    assert loaded["probabilities"].shape == probs.shape
    with pytest.raises(ValueError, match="patient IDs/order"):
        load_prediction_cache(cache, list(reversed(ids)))


def test_checkpoint_save_restore_roundtrip(tmp_path):
    model = torch.nn.Conv3d(1, 2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = tmp_path / "roundtrip.pt"
    save_checkpoint(path, model, optimizer, scaler, 3, {"seed":42}, {"best_epoch":2})
    with torch.no_grad():
        for parameter in model.parameters(): parameter.add_(1)
    epoch = load_checkpoint(path, model, optimizer, scaler, map_location="cpu")
    assert epoch == 3
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())


def test_multimodel_volume_cache_roundtrip(tmp_path):
    ids=["p1","p2"]
    labels=np.zeros((2,4,4,4),dtype=np.uint8)
    probs=np.full((2,4,4,4,4),.25,dtype=np.float32)
    path=tmp_path/"aligned.npz"
    class_probs=np.full((2,2),.5,dtype=np.float32)
    cosines=np.asarray([.1,.2],dtype=np.float32)
    save_multimodel_probability_cache(path,ids,{"model_a":probs,"model_b":probs},labels,
        {"split":"validation","grid":"64-cube"},
        classification_probabilities={"model_a":class_probs},paired_cosines={"model_a":cosines})
    loaded=load_multimodel_probability_cache(path,ids,{"split":"validation"})
    assert list(loaded["probabilities"])==["model_a","model_b"]
    assert loaded["probabilities"]["model_a"].shape==probs.shape
    assert loaded["classification_probabilities"]["model_a"].shape==(2,2)
    assert np.allclose(loaded["paired_cosines"]["model_a"],cosines)
    with pytest.raises(ValueError,match="patient IDs/order"):
        load_multimodel_probability_cache(path,list(reversed(ids)))


def test_swin_unetr_is_explicitly_optional():
    pytest.importorskip("monai")
    from neurolens.models.baselines import SwinUNETR3D
    model = SwinUNETR3D(in_channels=4, classes=4, feature_size=12, img_size=(32, 32, 32))
    with torch.inference_mode():
        logits = model(torch.randn(1, 4, 32, 32, 32))["segmentation_logits"]
    assert logits.shape == (1, 4, 32, 32, 32)
