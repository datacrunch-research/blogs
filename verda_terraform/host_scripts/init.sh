#!/bin/bash
# Terraform installation
wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(grep -oP '(?<=UBUNTU_CODENAME=).*' /etc/os-release || lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform


# Go installation: https://go.dev/dl/
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.25.5.linux-arm64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# verification
 go version

# Terraform custom provider
echo 'provider_installation {
  dev_overrides {
    "verda-cloud/verda" = "/mnt/cephfs/dc/rodri/IaC_blog/terraform-provider-verda"
  }
  direct {}
}' > $HOME/.terraformrc