(notebook-tutorial)=
# Notebook tutorials

## SevenNet-Polar: BEC and multitask training

The Polar notebooks use the public ZrO₂ example, train through the package
CLI, reload checkpoints through Python, and export held-out parity figures.
They exclude dielectric prediction.

| Notebook | Google Colab | Runtime |
|---|---|---|
| [BEC quickstart](https://github.com/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/01_bec_quickstart.ipynb) | [![Open in Google Colab]](https://colab.research.google.com/github/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/01_bec_quickstart.ipynb) | CPU / free T4 |
| [Multitask E/F/S/BEC](https://github.com/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/02_multitask_a100_flashtp.ipynb) | [![Open in Google Colab]](https://colab.research.google.com/github/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/02_multitask_a100_flashtp.ipynb) | A100 when available, optional FlashTP |

See the [Polar tutorial guide](https://github.com/AugustinLu/SevenNet-Polar/tree/main/tutorials/colab)
for validation records, dataset attribution, and split limitations. These are
preview links to the tutorials branch.

## Upstream SevenNet notebooks

If you want to learn how to use the `sevenn` Python library instead of the CLI command, please check out the notebook tutorials below.

Please note that the library is not fully documented and typical use cases can be covered by CLI.

| Notebooks | Google&nbsp;Colab | Descriptions |
|-----------|-------------------|--------------|
|[From scratch](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|We can learn how to train the SevenNet from scratch, predict energy, forces, and stress using the trained model, perform structure relaxation, and draw EOS curves.|
|[Fine-tuning](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|We can learn how to fine-tune the SevenNet and compare the results of the pretrained model with the fine-tuned model.|

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

Sometimes, the Colab environment may crash due to memory issues. If you have sufficient GPU resources in your local environment, we recommend downloading the tutorials from GitHub and running them on your machine.
```bash
git clone https://github.com/MDIL-SNU/sevennet_tutorial.git
```
