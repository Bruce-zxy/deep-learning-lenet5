import os
import struct
import numpy as np
import matplotlib.pyplot as plt
 
 
def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
 
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        # 'I'表示一个无符号整数，大小为四个字节
        # '>II'表示读取两个无符号整数，即8个字节
        labels = np.fromfile(lbpath, dtype=np.uint8)
 
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols =  struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
 
    return images, labels
 
 
X_train, y_train = load_mnist("data/", kind="train")
# X_test, y_test = load_mnist("data/", kind="t10k")
 
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

print(X_train, y_train)
 
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
 
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()