from .base_options import BaseOptions


class SpoutOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--spout_size', nargs = 2, type=int, default=[256, 256],
                                    help='Width and height of the spout receiver')
        parser.add_argument('--spout_in', type=str, default='spout_receiver_in',
                                    help='Spout receiving name - the name of the sender you want to receive')
        parser.add_argument('--spout_out', type=str, default='spout_receiver_out',
                                    help='Spout receiving name - the name of the sender you want to send')
        parser.add_argument('--window_size', nargs = 2, type=int, default=[10, 10],
                                    help='Width and height of the window')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=1024)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--output_dir', type=str, default='results',
                            help='Directory name to save the generated images')
        self.isTrain = False
        return parser
