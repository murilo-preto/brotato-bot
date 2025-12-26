import cv2
import numpy as np
from PIL import Image


def render_boxes(image, results=None, boxes=None, classes=None, class_names=None,
                 show_conf=True, conf_threshold=0.0, window_name='Detections', wait=1):
    """
    Draw bounding boxes on an image and show it with OpenCV.

    Args:
        image: BGR numpy array or PIL.Image (RGB). Returned image is BGR numpy.
        results: ultralytics Results list (uses results[0].boxes if provided).
        boxes: Nx4 array-like of xyxy boxes (overrides results if provided).
        classes: N array-like of class ids for `boxes` (required if `boxes` provided).
        class_names: dict or list mapping class id -> name.
        show_conf: whether to append confidence to the label (if available).
        conf_threshold: minimum confidence to draw a box (0.0 draws all).
        window_name: OpenCV window name used for display.
        wait: cv2.waitKey milliseconds. 0 blocks until key press, >0 waits that many ms, -1 does not call waitKey.

    Returns:
        Annotated BGR numpy array.
    """

    # Normalize input image to BGR numpy
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    xyxy = np.zeros((0, 4), dtype=float)
    confs = None
    cls_ids = np.array([], dtype=int)

    # Prefer explicit boxes/classes if provided
    if boxes is not None and classes is not None:
        xyxy = np.array(boxes, dtype=float)
        cls_ids = np.array(classes, dtype=int)
    elif results is not None and len(results) > 0 and getattr(results[0], 'boxes', None) is not None:
        b = results[0].boxes
        try:
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            cls_ids = b.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback if already numpy-like
            xyxy = np.array(b.xyxy)
            try:
                confs = np.array(b.conf)
            except Exception:
                confs = None
            try:
                cls_ids = np.array(b.cls).astype(int)
            except Exception:
                cls_ids = np.array([])

    # Drawing
    annotated = img.copy()

    colors = [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
        (128, 0, 128),
        (255, 165, 0)
    ]

    for i, box in enumerate(xyxy):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        conf = None if confs is None else float(confs[i])
        if conf is not None and conf < conf_threshold:
            continue

        cls_id = int(cls_ids[i]) if i < len(cls_ids) else None

        color = colors[(cls_id if cls_id is not None else i) % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = ''
        if class_names is not None and cls_id is not None:
            try:
                if isinstance(class_names, dict):
                    label = class_names.get(cls_id, str(cls_id))
                else:
                    label = class_names[cls_id]
            except Exception:
                label = str(cls_id)
        elif cls_id is not None:
            label = str(cls_id)

        if show_conf and conf is not None:
            label = f"{label} {conf:.2f}" if label else f"{conf:.2f}"

        if label:
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 5)
            cv2.rectangle(annotated,
                         (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0] + 10, label_y + 5),
                         color, -1)
            cv2.putText(annotated, label, (x1 + 5, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show image
    cv2.imshow(window_name, annotated)
    if wait is not None and wait >= 0:
        cv2.waitKey(int(wait))

    return annotated


__all__ = ['render_boxes']


def format_results(results, class_names=None, conf_threshold=0.0):
    """
    Convert ultralytics `results` into a list of dictionaries for each detection.

    Args:
        results: ultralytics Results list returned by `model.predict(...)`.
        class_names: optional mapping (dict or list) from class id to class name.
        conf_threshold: minimum confidence to include a detection.

    Returns:
        List[dict] where each dict contains keys: `class_name`, `class_id`,
        `x1`,`y1`,`x2`,`y2`,`cx`,`cy`,`w`,`h`,`conf`.
    """

    out = []
    if results is None or len(results) == 0:
        return out

    boxes = getattr(results[0], 'boxes', None)
    if boxes is None:
        return out

    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:
        xyxy = np.array(getattr(boxes, 'xyxy', []))

    try:
        confs = boxes.conf.cpu().numpy()
    except Exception:
        confs = np.array(getattr(boxes, 'conf', [])) if hasattr(boxes, 'conf') else None

    try:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        cls_ids = np.array(getattr(boxes, 'cls', []), dtype=int) if hasattr(boxes, 'cls') else np.array([], dtype=int)

    for i, box in enumerate(xyxy):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = map(float, box)
        conf = float(confs[i]) if confs is not None and i < len(confs) else None
        if conf is not None and conf < conf_threshold:
            continue

        cls_id = int(cls_ids[i]) if i < len(cls_ids) else None

        # Resolve class name
        name = None
        if class_names is not None and cls_id is not None:
            try:
                if isinstance(class_names, dict):
                    name = class_names.get(cls_id, str(cls_id))
                else:
                    name = class_names[cls_id]
            except Exception:
                name = str(cls_id)
        else:
            name = str(cls_id) if cls_id is not None else ''

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        out.append({
            'class_name': name,
            'class_id': cls_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h,
            'conf': conf
        })

    return out


__all__ = ['render_boxes', 'format_results']
