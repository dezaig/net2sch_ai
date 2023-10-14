import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
from torch.utils.data import DataLoader, Dataset, random_split

import os
import sys
import glob
import subprocess


def check_schematic_files(schematics_folder, netlist_folder):
    print(f"Checking schematic files in folder: {schematics_folder}")  # Display the folder being checked

    # Search for .sch files in the specified folder
    sch_files = glob.glob(os.path.join(schematics_folder, '*.sch'))

    num_files = len(sch_files)  # Number of .sch files
    if num_files == 0:
        print("No .sch files found. Please provide a folder containing .sch files.")
        sys.exit(1)

    print(f"Number of .sch files: {num_files}")

    # Create netlist folder if it doesn't exist
    if not os.path.exists(netlist_folder):
        os.makedirs(netlist_folder)

    for sch_file in sch_files:
        try:
            with open(sch_file, 'r') as f:
                # Run Qucs command to generate netlist
                base_name = os.path.basename(sch_file).split('.')[0]
                netlist_file = os.path.join(netlist_folder, f"{base_name}.cir")
                # Full path to Qucs executable
                qucs_executable = os.path.join(qucs_path, "bin\\qucs.exe")

                cmd = [qucs_executable, "--netlist", "-i", sch_file, "-o", netlist_file]
                subprocess.run(cmd, check=True)

                print(f"Successfully generated netlist for {sch_file} -> {netlist_file}")

        except Exception as e:
            print(f"Could not process file {sch_file} or generate netlist: {e}")
            sys.exit(1)





# Check if Qucs is installed
def check_qucs_installation(qucs_path):
    if not os.path.exists(qucs_path):
        print("Qucs path does not exist. Please provide a valid path.")
        sys.exit(1)
    # Try to run a Qucs command to check if it's working
    try:
        result = subprocess.run([os.path.join(qucs_path, "bin\\qucs.exe"), "-v"], capture_output=True, check=True, text=True)
        print(f"Qucs version check succeeded: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Qucs version check failed: {e.stderr}")
        sys.exit(1)

# Sanity check for dataset path and files
def check_dataset_path(dataset_path):
    print(f"Checking dataset path: {dataset_path}")  # Display the path being checked
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist. Please provide a valid path.")
        sys.exit(1)


class DummyQucsDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.rand(1), torch.rand(1)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, output_size)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs


def train_model(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    encoder = EncoderRNN(1, 16)
    decoder = DecoderRNN(16, 1)
    model = Seq2Seq(encoder, decoder)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x.view(1, 1, -1))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    qucs_path = "c:\qucs\qucs_0_0_19"
    dataset_path = "C:\github\\net2sch_ai\dataset\qucs\schematics"
    schematics_folder = "C:\github\\net2sch_ai\dataset\qucs\schematics"
    netlist_folder = "C:\github\\net2sch_ai\dataset\qucs\\netlist"


    # Perform sanity checks
    check_qucs_installation(qucs_path)
    check_dataset_path(dataset_path)

      # Perform sanity checks
    check_schematic_files(schematics_folder, netlist_folder)

    # Dummy function to represent the import of Qucs data
    def batch_export_qucs_schematics(qucs_path):
        print(f"Dummy function: Pretending to batch export from {qucs_path}")


    batch_export_qucs_schematics(qucs_path)

    # Generate a dummy dataset with 100 samples
    dataset = DummyQucsDataset(100)

    # 80-20 train-test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_model(train_dataset)
