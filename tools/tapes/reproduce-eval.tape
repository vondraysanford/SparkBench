# tools/tapes/reproduce-eval.tape — the Phase 7 "reproduce the eval in 30 minutes" demo as a script.
#
# Render:   vhs tools/tapes/reproduce-eval.tape        (brew install vhs ttyd ffmpeg)
# Why VHS:  the demo is typed by a script, so the GIF is regenerated after every change and
#           can never drift from what the README claims. Anyone can re-render it.
#
# Edit the commands below once experiments/01-finetune-vs-frontier/ has its reproduce script.

Output experiments/01-finetune-vs-frontier/evidence/phase-7/reproduce-eval.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 20

Type "git clone https://github.com/vondraysanford/sparkbench && cd sparkbench"
Enter
Sleep 2s

Type "make reproduce-eval   # pinned deps, one script, fixture download"
Enter
Sleep 20s

Type "cat experiments/01-finetune-vs-frontier/eval/results/summary.md"
Enter
Sleep 6s
