from trainer import Trainer
from tester import Tester
from dataset import Dataset, Params

if __name__ == "__main__":
    args = Params()
    dataset = Dataset(args.dataset)
    #
    print("~~~~ Training ~~~~")
    trainer = Trainer(dataset, args)
    trainer.train()

    print("~~~~ Select best epoch on validation set ~~~~")
    dataset = Dataset(args.dataset)

    checkpoint_file = "models/" + args.dataset + "/model.chkpnt"
    tester = Tester(dataset, checkpoint_file, "dev")
    tester.test()

    tester = Tester(dataset, checkpoint_file, "test")
    tester.test()
