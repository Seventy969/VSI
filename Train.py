"""
================================================================================
  VSI SVM Training Script
================================================================================
  1. Reads face training images from  Faces/train/<PersonName>/*.jpg
  2. Detects & aligns faces with MTCNN
  3. FaceNet 512-D L2-normalised embeddings
  4. Trains a linear SVC with probability estimates (probability=True)
  5. Outputs: OCV_data/FACE_SVM.pkl > SVM classifier
              OCV_data/FACE_svm_embeddings.npy > per-person mean embedding for cosine guard in main app
                                               > greatly reduces false-positive rate on unknown faces

  Directory structure expected:
      Faces/
      └── train/
          ├── Your Name/
          │   ├── 1.jpg
          │   └── 2.jpg
          ├── Someone Name/
          │   └── 1.jpg
          └── ...

  Requirements:
  ─────────────────────────────────────────────────
      pip install PyQt5 | For building the graphical user interface (GUI).
      pip install opencv-contrib-python | For advanced image processing and additional OpenCV modules.
      pip install ultralytics | YOLOv8 face detection
      pip install facenet-pytorch | Face detection + embedding
      pip install mediapipe==0.10.20 | Hand, face, and body tracking
      pip install scikit-learn | SVM classifier for face recognition
      pip install joblib | Save/load trained models efficiently
      pip install "numpy<2.0" | For numerical computations and array manipulation.
                                Warning: numpy version 2.0 and above will fails in PYtorch
                                apps!
      pip install onnxruntime requests | Emotion Detection
      pip install omegaconf | 
      pip install pyside6 | Qt for python
      pip install torch | Deep learning backend
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      pip install icecream | For easy and advanced debugging/logging.
      pip install PyQt5Designer | For development of ui file. “pip install PyQt5 pyqt5-tools” is alternative to package 1 and 5
================================================================================
"""
import os, cv2, numpy as np, torch, joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# ── CONFIG — edit these ───────────────────────────────────────────────────────
TRAIN_DIR  = r"C:\Users\User\Desktop\VSI\Faces\train" # Path of training images (one sub-folder per person)
OUTPUT_PKL = r"OCV_data/FACE_SVM.pkl"                 # Output SVM model path
PEOPLE     = []                                       # Leave [] empty = to auto-detect all sub-folders
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")
print("Loading MTCNN & FaceNet... ...")
mtcnn  = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
print("Models ready.\n")


def get_people(train_dir, plist):
    # Return list of person names — from argument or auto-scan the folder.
    if plist:
        return plist
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Directory not found: {train_dir}")
    return sorted(d for d in os.listdir(train_dir)
                  if os.path.isdir(os.path.join(train_dir, d)))


def embed(img_bgr):
    # Run MTCNN & FaceNet, return L2-normalised 512-D embedding.
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        return None
    with torch.no_grad():
        e = resnet(face.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9) # L2 normalise


def collect(people, train_dir):
    # Walk each person's folder to detect faces.
    embs, labs = [], []
    for p in people:
        d = os.path.join(train_dir, p)
        if not os.path.isdir(d):
            print(f" ⚠ Directory not found: {d} | SKIP... ..."); continue
        imgs = [f for f in os.listdir(d)
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        print(f"  [{p}] {len(imgs)} images found")
        ok = 0
        for f in imgs:
            img = cv2.imread(os.path.join(d, f))
            if img is None:
                print(f" ⚠ Cannot read: {f}"); continue
            e = embed(img)
            if e is None: 
                print(f" ⚠ No face detected: {f}"); continue
            embs.append(e); labs.append(p); ok += 1
        print(f"         -> {ok} embeddings OK")
    return np.array(embs), np.array(labs)


def build_cache(embs, labs, people):
    # Compute per-person mean L2-normalised embedding.
    # Saved alongside the SVM for the cosine guard in the main app.
    cache = {}
    for p in people:
        mask = labs == p
        if not mask.any(): continue
        m = embs[mask].mean(0)
        cache[p] = m / (np.linalg.norm(m) + 1e-9)
    return cache


def main():
    print("=" * 30)
    print("  VSI SVM Training ")
    print("=" * 30)
    people = get_people(TRAIN_DIR, PEOPLE)
    if not people:
        raise RuntimeError(f"No person folders in: {TRAIN_DIR}")
    print(f"\nIdentities: {people}\n")
    embs, labs = collect(people, TRAIN_DIR)
    if len(embs) == 0:
        raise RuntimeError("No embeddings collected.")
    print(f"\nTotal: {len(embs)} embeddings, {len(set(labs))} identities")

    # Train SVM
    print("\nTraining SVC…")
    clf = SVC(kernel="linear", probability=True, C=1.0)
    clf.fit(embs, labs)

    # Cross-validation
    n = min(5, len(embs))
    if n >= 2:
        sc = cross_val_score(clf, embs, labs, cv=n, scoring="accuracy")
        print(f"C-Val accuracy: {sc.mean():.1%}  (±{sc.std():.1%})")
    else:
        print("(Not enough samples for cross-validation)")

    # Save SVM
    os.makedirs(os.path.dirname(OUTPUT_PKL) or ".", exist_ok=True)
    joblib.dump(clf, OUTPUT_PKL)
    print(f"\nSVM saved → {OUTPUT_PKL}")

    # Save embedding cache (used by cosine guard in main app)
    cache = build_cache(embs, labs, people)
    np.save(OUTPUT_PKL.replace(".pkl","_embeddings.npy"), cache)
    print(f"Cache saved → {OUTPUT_PKL.replace('.pkl','_embeddings.npy')}")
    print(f"Identities: {list(cache.keys())}")
    print(f"\nDone.  Load {OUTPUT_PKL} in the main app via 'Load SVM Model'.")


if __name__ == "__main__":
    main()
