# Advanced Deep Learning Lecture Slides

This directory contains materials for an **advanced lecture** on deep learning topics. Its main topics are modern deep learning with transformers and related models. Main topic will be NLP applications and a little computer vision.

## Slide Files and Overview

The slides will be made in latex with beamer. The high level file is adl.tex, and each subtopic will get its own tex file, which will be included from adl.tex
- intro.tex: Introduction, administration etc.
- transformer.tex: Transformer architecture and attention
- gpt2.tex: Training a GPT2 like model
- rnn.tex: RNN, LSTM, GRU
- mamba.tex: Mamba 1/2. Use material from Tri Dao's blog https://tridao.me/blog/2024/mamba2-part1-model/ and the blogpost https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
- rl.tex: Reinforcement learning: Policy gradient, REINFORCE, PPO, GRPO, DPO for training LLMs.
- parallelism.tex is a hub file for multiple sub-files. Material is based on the ultra scale playbook from https://nanotron-ultrascale-playbook.static.hf.space/index.html

## Style

- Do not use em-dash
- Use single column layout by default.
- Use two-column layout when appropriate, e.g. sometimes when using illustrations, or when you have a very information dense slides.
- Use tikz images for illustration when useful. All tikz libraries necessary are already loaded.
- Structure text as much as possible and be succinct with text. Use itemize, description etc. whenever possible and appropriate
- Use algorithm2e environments to write pseudo-code. Add a title on top of each algorithm.

## References

- References are stored in references.bib.
- Cite articles, blog posts etc. whenever they come up. Cite them mostly once, not each time they are referenced.

## Notes

- Do not compile except when requested. Automatic compilation is turned on in another process.
- When you try to make some illustration into a tikz file, first write the illustration into a temporary file, compile and compare against the original illustration. Improve until it looks good enough. Take care of crossing edges, ugly edges start and ends, overlaps etc.
- Do not have claude in the commit messages
