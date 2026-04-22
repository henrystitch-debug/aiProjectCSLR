import re
import matplotlib.pyplot as plt

log_file = "log.txt"

epochs = []
losses = []
wers = []

current_epoch = None
current_loss = None

with open(log_file, "r") as f:
    for line in f:
        # Match epoch start
        epoch_match = re.search(r'Epoch:\s*(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Match mean loss
        loss_match = re.search(r'Mean training loss:\s*([0-9.]+)', line)
        if loss_match:
            current_loss = float(loss_match.group(1))

        # Match WER
        wer_match = re.search(r'Dev WER:\s*([0-9.]+)%', line)
        if wer_match and current_epoch is not None and current_loss is not None:
            wer = float(wer_match.group(1))

            epochs.append(current_epoch)
            losses.append(current_loss)
            wers.append(wer)

            # reset to avoid duplication
            current_epoch = None
            current_loss = None

# Convert WER to accuracy if desired
accuracies = [100 - w for w in wers]

# Filter epochs % 5 == 0
epochs_mod5 = [e for e in epochs if e % 5 == 0]
losses_mod5 = [losses[i] for i, e in enumerate(epochs) if e % 5 == 0]
acc_mod5 = [accuracies[i] for i, e in enumerate(epochs) if e % 5 == 0]

# -----------------
# Plot Loss
# -----------------
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label="All epochs")
plt.scatter(epochs_mod5, losses_mod5, color='red', label="Epoch % 5 == 0")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.grid()
plt.show()

# -----------------
# Plot Accuracy
# -----------------
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracies, label="All epochs")
plt.scatter(epochs_mod5, acc_mod5, color='red', label="Epoch % 5 == 0")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epoch (100 - WER)")
plt.legend()
plt.grid()
plt.show()