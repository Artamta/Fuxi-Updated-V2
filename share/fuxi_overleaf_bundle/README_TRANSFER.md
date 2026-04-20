Fuxi Overleaf Bundle

What is here:
- `main.tex`: upload-ready LaTeX entrypoint for Overleaf
- `report.tex`: original report source copied from `report/report.tex`
- root-level PNG files: figures referenced by the current report
- `extras/prof/`: professor-report helpers, GIFs, and summary plots
- `extras/results_new/`: selected forecast PNGs and full metrics folders for `emb768` and `emb1024`
- `extras/pretrain_scaling/`: config, history, and plot files for the embed-dimension sweep

Recommended transfer from your Mac:

```bash
scp -r raj.ayush@gpu2:/home/raj.ayush/fuxi-final/fuxi_new/share/fuxi_overleaf_bundle /Users/ayush/Desktop/Fuxi-diagrams/
```

If you prefer a single archive, use the zip file created beside this folder:

```bash
scp raj.ayush@gpu2:/home/raj.ayush/fuxi-final/fuxi_new/share/fuxi_overleaf_bundle.zip /Users/ayush/Desktop/Fuxi-diagrams/
```

Overleaf notes:
- Upload the folder contents or the zip file.
- Set `main.tex` as the main file if Overleaf does not auto-detect it.
- The current report figures are already placed next to `main.tex`, so it should compile without path edits on Overleaf.
