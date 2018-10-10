import load_dataset as ld
import training as tr
import testing as te


def main():
    # ld.load_all()
    # tr.start_training()
    text = input("Input your email title : ")
    # te.start_testing("You've received messages from elderdasilva")

    te.start_testing(str(text))


if __name__ == '__main__':
    main()
