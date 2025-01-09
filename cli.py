
def add_all_parsers(parser):
    _add_loss_parser(parser)
    _add_training_parser(parser)
    _add_misc_parser(parser)

def _add_loss_parser(parser):
    group_loss = parser.add_argument_group('Loss parameters')
    group_loss.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay parameter')
    group_loss.add_argument('--lr', type=float, default=1e-2, help='learning rate to use')
    group_loss.add_argument("--momentum", type=float, default=0.9, help='momentum"')


def _add_training_parser(parser):
    group_training = parser.add_argument_group('Training parameters')
    group_training.add_argument('--batch_size', '-b', type=int, default=4)
    group_training.add_argument('--num_epochs', '-n', type=int, default=1000)
    group_training.add_argument('--image_size', type=int, default=320)
    group_training.add_argument('--model_name', type=str, default="model_4")
    group_training.add_argument('--num_classes', type=int, default=1081)

def _add_misc_parser(parser):
    group_misc = parser.add_argument_group('Miscellaneous parameters')
    group_misc.add_argument('--num_workers', type=int, default=4)
    group_misc.add_argument('--root', default="plantnet_300K", help='location of the train val and test directories')
    group_misc.add_argument("--checkpoint_dir", "-c", type=str, default="trained_models")
    group_misc.add_argument("--checkpoint", "-p", type=str, default=None)
    group_misc.add_argument("--tensorboard_dir", "-t", type=str, default="tensorboard")
