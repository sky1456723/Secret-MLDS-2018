def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size','-b', type = int, default = 32,
                        help = 'Specify the batch size for training')
    parser.add_argument('--episode', '-ep', type = int, default = 1000,
                        help = "Specify how many ep play in an epoch")
    parser.add_argument('--learning_rate','-lr', type = float, default = 0.0001,
                        help = 'Specify the learning rate for training')
    parser.add_argument('--gamma', type = float, default = 0.99,
                        help = 'Specify the discount factor of reward')
    parser.add_argument('--baseline', type = float, default = 0,
                        help = "Specify the baseline for reward function")
    parser.add_argument('--optim', type = str, default = "Adam",
                        help = "Specify the optimizer")
    parser.add_argument('--PPO', action='store_true', default = False,
                        help = "Specify to use PPO or not")
    
    parser.add_argument('--model_name', type = str, default = "Default_model.pkl",
                        help = "Specify the name of the model")
    
    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help='load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help='create a new model')
    mutex.add_argument('--remove_model', '-r', action='store_true', help='remove a model')
    
    return parser
