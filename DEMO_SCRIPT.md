
# MANZAR — 5-MINUTE DEMO SCRIPT
================================

## BEFORE DEMO (5 min setup)
1. Start backend: cd manzar-backend && source venv312/bin/activate && python -m uvicorn app.main:app --port 8000
2. Open browser: http://localhost:8000/MANZAR.html
3. Verify backend running: curl http://localhost:8000/health

## DEMO FLOW (5 minutes)

### MINUTE 1: The Problem (30 sec)
"Every year, 1.35 million people die in road accidents. In India alone, 1.5 lakh accidents 
go poorly documented — evidence disappears within minutes, witnesses leave, and FIRs are 
filed days later with incomplete data. What if we could capture the entire accident scene 
in 5 minutes using multiple phones?"

### MINUTE 2: Create Session (30 sec)
[Click "Create Session"]
- Fill: Location = "NH-48, Near Hero Honda Chowk, Gurugram"  
- Fill: Officer = "Inspector Rajesh Kumar"
- Fill: FIR Reference = "FIR-2026-08-GGM-0342"
- Click "Create Session & Generate QR Code"

**Talking point:** "Officer creates a session, gets a QR code. Bystanders, witnesses, 
other drivers — everyone scans and captures from their angle."

### MINUTE 3: Multi-Phone Capture (1 min)
[Show QR code screen]
**Talking point:** "3 phones scan the QR code. Each captures from a different angle:
- Phone 1: Front view — shows vehicle damage
- Phone 2: Right side — shows skid marks  
- Phone 3: Rear view — shows debris field
- Phone 1 again: Left side — shows impact point"

[Upload photos from each angle]
- Watch: Detection runs on each photo (bounding boxes appear)
- Watch: Coverage meter fills: 1/4 → 2/4 → 3/4 → 4/4
- Watch: Cross-angle verification triggers

**Talking point:** "YOLOv8 runs on each photo instantly — detecting vehicle damage, skid marks, 
debris, and impact points. The system knows which angles are covered and prompts for missing ones."

### MINUTE 4: Dashboard & Evidence (1.5 min)
[Click "Go to Dashboard"]

**Point out:**
1. **Photo gallery with bounding boxes** — "See the colored boxes around evidence"
2. **Investigation timeline** — "Every event is timestamped — chain of custody"
3. **Cross-angle score: 100%** — "Same evidence detected by multiple cameras"
4. **Case details** — "GPS, officer name, case number — all linked"

[Click "Ask MANZAR"]
Ask: "What evidence was detected?"
→ Forensic expert AI responds with structured analysis

Ask: "Show the cross-angle verification"
→ Shows multi-camera corroboration

Ask: "Who is at fault?"  
→ AI refuses: "Fault requires comprehensive investigation"

**Talking point:** "Gemini-powered forensic AI — speaks like an expert witness, never 
speculates on fault, always cites evidence."

### MINUTE 5: Report & Export (1 min)
[Click "Generate Report"]

**Show:**
- Case header with officer, GPS, FIR number
- Evidence findings table with severity
- Forensic analysis (speed estimation if skid marks)
- Cross-angle verification score
- Chain of custody with SHA-256 hashes
- Investigation timeline

[Click "Export as PDF"]
→ Downloads professional police-style PDF

**Talking point:** "In 5 minutes: multi-angle evidence captured, AI-analyzed, cross-verified, 
and documented in a professional forensic report. What used to take 3 days now takes 5 minutes."

### CLOSING (30 sec)
"MANZAR — Multi-phone Evidence Capture, YOLOv8 Detection, Cross-Angle Verification, 
Gemini AI Analysis, and Auto-Generated Forensic Reports. From accident scene to FIR 
in 5 minutes."

---

## KEY DEMO POINTS (hit these for judges)

1. **Multi-phone collaboration** — QR-based, no app install, works on any phone
2. **Coverage-aware capture** — system ensures 360° documentation
3. **AI detection** — YOLOv8 trained on accident evidence (4 classes)
4. **Cross-angle corroboration** — evidence seen by 2+ cameras = higher reliability
5. **Forensic AI chatbot** — Gemini-powered expert that refuses to speculate on fault
6. **Tamper-proof chain of custody** — SHA-256 hashes, timestamps, device IDs
7. **Auto-generated PDF report** — police-ready format with all evidence

## FAQ ANSWERS (if judges ask)

Q: "How does cross-angle verification work?"
A: "Same evidence type detected from different cameras is grouped. If 2+ cameras see the 
same type of evidence (e.g., vehicle damage), it's marked as CORROBORATED. Score = 
corroborated items / total evidence types."

Q: "What about privacy?"
A: "All processing happens locally. No cloud storage. Photos are hashed for tamper detection. 
Chain of custody follows digital evidence standards."

Q: "How fast is the 3D reconstruction?"
A: "COLMAP runs locally — 15-25 min for full reconstruction. For demo, we can upload 
pre-computed mesh from Kaggle GPU. The reconstruction adds 3D spatial context to 2D photos."

Q: "Can this work without internet?"
A: "Yes — the backend runs on a laptop. Phones connect via local WiFi or hotspot. 
Only the Gemini AI chatbot needs internet — everything else works offline."

## DEMO PHOTOS NEEDED

For maximum impact, use real accident scene photos showing:
- Clear vehicle damage (dented bumper, broken windshield)
- Skid marks on road surface
- Debris scattered on road (glass, metal fragments)
- Point of impact (scrape marks, paint transfer)

Sources for free accident photos:
- Roboflow Universe: search "vehicle damage", "car accident"
- Kaggle: "car damage detection dataset"
- Google Images: "car accident scene" (use Creative Commons filter)

TIP: Download 8-12 photos showing all 4 evidence types.
Upload 4 from each "phone" (angle) for best cross-angle results.
