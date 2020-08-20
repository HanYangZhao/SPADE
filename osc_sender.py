from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--osc_receive_port",type=int, default=5005, help="The port to listen on")
  parser.add_argument("--osc_send_port",type=int, default=5006, help="The port to listen on")
  parser.add_argument("--osc_send_ip",type=str, default="127.0.0.1", help="The port to listen on")
  parser.add_argument("--osc_channel",type=str, default="/spadeGAN", help="OSC channel")
  parser.add_argument("--file_path",type=str, help="path to test file")
  args = parser.parse_args()

  dispatcher = dispatcher.Dispatcher()
  dispatcher.map(args.osc_channel, print)
  client = udp_client.SimpleUDPClient(args.osc_send_ip, args.osc_send_port)

  client.send_message(args.osc_channel, args.file_path)

  server = osc_server.ThreadingOSCUDPServer(
      ("0.0.0.0", args.osc_receive_port), dispatcher)
  print("Serving on {}".format(server.server_address))

  server.serve_forever()