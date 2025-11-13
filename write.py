import os


def getfiles():
    filenames = os.listdir('datasets/UIEB/sr_16_256')
    print(filenames)
    return filenames


if __name__ == '__main__':

    # Ensure 'train.txt' exists or create it
    # if not os.path.exists('data/UIEB/test.txt'):
    #     with open('C:/Code/Python/UIE/UIEC2Net/data/UIEB/test.txt', 'w') as f:
    #         pass

    a = getfiles()
    l = len(a)
    with open("data/UIEB/test.txt", "w") as f:
        for i in range(l):
            print(a[i])
            x = a[i]
            f.write(x)
            f.write('\n')

    print("File writing completed successfully.")
