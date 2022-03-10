import argparse
import numpy as np


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--file_name", help="input: set a path to the accuuracy file")

        args = parser.parse_args()
        
        epochs = 350
        test_accuracy = np.zeros((epochs,10))

        with open(args.file_name, 'r') as filehandle:
            filecontents = filehandle.readlines()
            index = 0
            col = 0
            for line in filecontents:
                t_acc = line[:-1]
                test_accuracy[index][col] = float(t_acc)
                index += 1
                if index == epochs:
                    index = 0
                    col += 1
                    if col == 10:
                        break
  
        ave_accuracy = np.mean(test_accuracy, axis=1)
        std_accuracy = np.std(test_accuracy, axis=1)
        #print(test_accuracy)
        #print(ave_accuracy)
        #print(std_accuracy)

        max_ave_accuracy = np.max(ave_accuracy)
        max_ind = np.argmax(ave_accuracy)
        max_std_accuracy = std_accuracy[max_ind]

        print('test accuracy / mean(std): {0:.5f}({1:.5f})'.format(max_ave_accuracy, max_std_accuracy))

    except IOError as e:
        print(e)
