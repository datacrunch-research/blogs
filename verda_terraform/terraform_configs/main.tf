terraform {
  required_providers {
    verda = {
      source = "verda-cloud/verda"
      version = "0.0.1"
    }
  }
}

provider "verda" {}



resource "verda_ssh_key" "tf_ssh" {
  name       = "tf-ssh"
  public_key = file("/mnt/cephfs/dc/rodri/blogs/verda_terraform/rodrimacos_25519.pub")
}

resource "verda_startup_script" "init_vm" {
  name   = "init-vm"
  script = file("vm_scripts/vm_init.sh")
}

resource "verda_volume" "tf_volume" {
  name     = "terraform-volume"
  size     = 2000  # Size in GB
  type     = "NVMe"
  location = "FIN-03"
}

resource "verda_instance" "terraform-sglang" {
  instance_type = "4B200.120V"
  image         = "ubuntu-24.04-cuda-13.0-open-docker"
  hostname      = "terraform-sglang"
  description   = "Example instance"
  location      = "FIN-03"

  ssh_key_ids = [verda_ssh_key.tf_ssh.id]
  startup_script_id = verda_startup_script.init_vm.id
  existing_volumes = [verda_volume.tf_volume.id]
}

# Outputs
output "instance_ip" {
  value = verda_instance.terraform-sglang.ip
}

output "instance_status" {
  value = verda_instance.terraform-sglang.status
}