import re
import matplotlib.pyplot as plt
import matplotlib as mpl

# -------------------------
# LaTeX-style configuration
# -------------------------
mpl.rcParams.update({
    "text.usetex": False,          # set True if you have LaTeX installed
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
})

log_file = "log.txt"

epochs, losses, wers = [], [], []

current_epoch = None
current_loss = None

# -------------------------
# Parse log
# -------------------------
with open(log_file, "r") as f:
    for line in f:
        epoch_match = re.search(r"Epoch:\s*(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        loss_match = re.search(r"Mean training loss:\s*([0-9.]+)", line)
        if loss_match:
            current_loss = float(loss_match.group(1).rstrip("."))

        wer_match = re.search(r"Dev WER:\s*([0-9.]+)%", line)
        if wer_match and current_epoch is not None and current_loss is not None:
            epochs.append(current_epoch)
            losses.append(current_loss)
            wers.append(float(wer_match.group(1)))

            current_epoch = None
            current_loss = None

accuracies = [100 - w for w in wers]

# -------------------------
# Filter checkpoints (epoch % 5 == 0)
# -------------------------
epochs_mod5 = [e for e in epochs if e % 5 == 0]
loss_mod5 = [losses[i] for i, e in enumerate(epochs) if e % 5 == 0]
acc_mod5 = [accuracies[i] for i, e in enumerate(epochs) if e % 5 == 0]

# -------------------------
# Figure 1: Loss
# -------------------------
plt.figure(figsize=(6.5, 4))

plt.plot(epochs, losses, label="Training Loss", color="black")
plt.scatter(epochs_mod5, loss_mod5, color="red", s=50, label="Checkpoint (mod 5)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("loss_curve_latex.pdf")
plt.savefig("loss_curve_latex.png", dpi=300)
plt.show()

# -------------------------
# Figure 2: Accuracy
# -------------------------
plt.figure(figsize=(6.5, 4))

plt.plot(epochs, accuracies, label="Dev Accuracy (100 - WER)", color="black")
plt.scatter(epochs_mod5, acc_mod5, color="blue", s=50, label="Checkpoint (mod 5)")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy vs Epoch")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("accuracy_curve_latex.pdf")
plt.savefig("accuracy_curve_latex.png", dpi=300)
plt.show()