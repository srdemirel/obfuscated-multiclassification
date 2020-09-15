import os
import numpy as np
import matplotlib.pyplot as plt

epoch = []
loss = []
acc = []
val_loss = []
val_acc = []

rows = [row.rstrip('\n') for row in open("./results/lstm_results.txt")]

for idx, row in enumerate(rows):
    epoch.append(idx)
    loss.append(np.float64(row.split(':')[1].split('-')[0]))
    acc.append(100*np.float64(row.split(':')[2].split('-')[0]))
    val_loss.append(np.float64(row.split(':')[3].split('-')[0]))
    val_acc.append(100*np.float64(row.split(':')[4].split('-')[0]))

plt.plot(epoch, loss, 'bo', label='Training loss')
plt.plot(epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./results/lstm_loss_fig.png',dpi=300, facecolor='w', edgecolor='w')
plt.show()

plt.clf()
plt.plot(epoch, acc, 'bo', label='Training accuracy')
plt.plot(epoch, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('./results/lstm_acc_fig.png',dpi=300, facecolor='w', edgecolor='w')
plt.show()
