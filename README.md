# Over the Shoulder 👀

**An AI study tutor that works on real paper.**

Over the Shoulder watches you solve math and physics problems by hand, finds the first step you got wrong, and marks it in red directly on your page with a short hint, so you fix the mistake yourself instead of being handed the answer.

<!-- Replace with your demo GIF/video -->
![Over the Shoulder demo](assets/demo.gif)

---

## How it works

```
overhead webcam ──► detect problems + lines (Roboflow RF-DETR)
                          │
                          ▼
              crop each problem, number its steps
                          │
                          ▼
         vision-language model checks the reasoning
         → { "first_wrong_step": 3, "hint": "..." }
                          │
                          ▼
   camera → projector homography ──► red mark + hint projected on the paper
```

1. **Capture**: An overhead webcam watches the page. Over the Shoulder only runs when the page has been still for about 1–2 seconds, so it never interrupts you mid-stroke.
2. **Detect**: A custom RF-DETR model trained on Roboflow finds each `problem` block and each `step` (line of work).
3. **Reason**: Each problem crop, with its steps numbered, is sent to a vision-language model through a Roboflow Workflow. The model returns the first incorrect step and a one-line hint, never the full solution.
4. **Project**: A one-time calibration (projected checkerboard + `cv2.findHomography`) maps camera pixels to projector pixels, so the red mark lands exactly on the wrong line.

No projector? Over the Shoulder also runs in **screen mode** and draws the marks on the live camera feed.

## Features

- ✍️ Works on real handwriting: no tablet, no app, just paper
- 📐 Math and physics, from algebra and calculus to kinematics and circuits
- 🎯 Pinpoints the *first* wrong step, not just "this answer is wrong"
- 💡 Gives hints, not answers
- 🔦 Projects feedback directly onto the page (or onto the screen)

## Tech stack

| Part | Tool |
|---|---|
| Object detection | Roboflow, RF-DETR |
| Pipeline | Roboflow Workflows (detection → crop → VLM) |
| Reasoning | Vision-language model (via Workflow VLM block) |
| Vision / projection | Python, OpenCV |
| Hardware | USB webcam, any HDMI projector |

## Dataset

- Overhead webcam frames of handwritten math and physics work, with right and deliberately wrong solutions, varied pens and lighting
- Bootstrapped with public Roboflow Universe datasets ([math_detect](https://universe.roboflow.com/ivanv2009-cmgfs/math_detect), [Line Text Detection](https://universe.roboflow.com/edm/line-text-detection)), remapped to the `step` class
- Classes: `problem`, `step`

<!-- Add your dataset/model link and mAP once trained -->
**Model:** RF-DETR, mAP@50: _TBD_ · [Roboflow project](#)

## Setup

```bash
git clone https://github.com/<your-username>/over-the-shoulder.git
cd over-the-shoulder
pip install -r requirements.txt
```

Create a `.env` file:

```
ROBOFLOW_API_KEY=your_key
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_WORKFLOW_ID=your_workflow_id
```

## Usage

**Screen mode (no projector):**
```bash
python main.py --mode screen
```

**Projector mode:**
1. Mount the webcam and projector above your desk, both pointing down at the page.
2. Connect the projector over HDMI and set your display to **extend** (not mirror).
3. Calibrate once:
   ```bash
   python calibrate.py
   ```
4. Run:
   ```bash
   python main.py --mode projector
   ```

**Tips:** Use matte white paper, write in dark pen or thick pencil, and keep the room a little dim so the projected marks pop.

## Roadmap

- [ ] Problem + step detection (RF-DETR)
- [ ] VLM step checking via Roboflow Workflows
- [ ] Projector calibration + on-paper marks
- [ ] Diagram support (free-body diagrams, circuits)
- [ ] Voice hints
- [ ] Session history: which kinds of mistakes you make most

## Author

Built by **Aurick Anwar**, Engineering @ McMaster University.
