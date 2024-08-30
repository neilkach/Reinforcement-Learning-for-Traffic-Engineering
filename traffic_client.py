import socket
import subprocess
import re
import time

def run_iperf(target_ip='10.20.0.212'):
    # Run iperf command and capture the output
    result = subprocess.run(['iperf', '-c', target_ip, '-t', '0.1', '-i', '0.1'], capture_output=True, text=True)
    return result.stdout

def parse_iperf_output(output):
    # Extract packet size, loss, and RTT from the iperf output
    packet_size = re.search(r'(\d+)\s+MBytes', output)
    packet_loss = re.search(r'(\d+)%\s+packet\s+loss', output)
    rtt = re.search(r'(\d+.\d+)\s+ms', output)

    print(packet_size, packet_loss, rtt)

    packet_size = packet_size.group(1) if packet_size else 'N/A'
    packet_loss = packet_loss.group(1) if packet_loss else 'N/A'
    rtt = rtt.group(1) if rtt else 'N/A'

    return packet_size, packet_loss, rtt

def main(target_ip, server_ip, port, iterations, delay):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    
    for i in range(iterations):
        iperf_output = run_iperf(target_ip)
        print(iperf_output)
        packet_size, packet_loss, rtt = parse_iperf_output(iperf_output)
        data = f'Packet Size: {packet_size} Mbytes, Packet Loss: {packet_loss}%, RTT: {rtt} ms'
        print(data)
        client_socket.sendall(data.encode())
        time.sleep(delay)  # Wait for a specified delay before the next iteration
    
    client_socket.close()

if __name__ == '__main__':
    target_ip = '10.20.0.212'
    server_ip = '10.10.0.201'
    port = 7008
    iterations = 300  # Number of iterations
    delay = 1  # Delay between iterations in seconds
    main(target_ip, server_ip, port, iterations, delay)