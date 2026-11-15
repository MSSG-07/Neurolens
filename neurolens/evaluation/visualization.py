"""Optional plotting utilities for CAM, overlays, embeddings, and curves."""
import numpy as np


def grad_cam_2d(model, image, target_class=None):
    """Grad-CAM for Encoder2D's final feature map; returns a normalized 2D map."""
    encoder=model.encoder2d if hasattr(model,"encoder2d") else model
    activation=[]; gradient=[]
    def capture(module, inputs, output):
        activation.append(output)
        output.register_hook(lambda g: gradient.append(g))
    handle=encoder.features.register_forward_hook(capture)
    try:
        out=encoder(image); cls=int(out["logits"].argmax(1)[0]) if target_class is None else target_class
        encoder.zero_grad(set_to_none=True); out["logits"][:,cls].sum().backward()
        if not gradient: raise RuntimeError("Grad-CAM gradient hook was not captured")
        weights=gradient[0].mean((2,3),keepdim=True); cam=(weights*activation[0]).sum(1,keepdim=True).relu()
        cam=torch.nn.functional.interpolate(cam,size=image.shape[-2:],mode="bilinear",align_corners=False)
        return cam[0,0].detach().cpu().numpy()/(cam.max().item()+1e-8)
    finally: handle.remove()


def save_segmentation_overlay(image, prediction, path, slice_index):
    import matplotlib.pyplot as plt
    base=np.asarray(image); seg=np.asarray(prediction)
    fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(base,cmap="gray"); ax.imshow(np.ma.masked_where(seg==0,seg),alpha=.45,vmin=1,vmax=3,cmap="viridis"); ax.axis("off"); fig.savefig(path,bbox_inches="tight"); plt.close(fig)


def plot_training_curves(csv_path, output_path):
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("Training CSV is empty")
    cols=[c for c in rows[0] if c.endswith("total")]
    fig,ax=plt.subplots()
    for col in cols: ax.plot([float(r["epoch"]) for r in rows],[float(r[col]) for r in rows],label=col)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); fig.savefig(output_path,bbox_inches="tight"); plt.close(fig)


def plot_embeddings(embedding2d, embedding3d, path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    x=np.concatenate((np.asarray(embedding2d),np.asarray(embedding3d))); y=TSNE(n_components=2,random_state=42,perplexity=min(30,len(x)-1)).fit_transform(x)
    plt.scatter(y[:len(embedding2d),0],y[:len(embedding2d),1],label="2D"); plt.scatter(y[len(embedding2d):,0],y[len(embedding2d):,1],label="3D"); plt.legend(); plt.savefig(path); plt.close()
