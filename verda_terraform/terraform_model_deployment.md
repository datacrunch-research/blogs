# Deploying DeepSeek R1 NVFP4 using Verda Terraform provider

This guide shows how to deploy an SGLang server on Verda using Terraform, including NVMe storage, Docker caching on local disk, and a reproducible benchmark run.

By the end, we’ll have a single Terraform config that provisions the infrastructure and a startup script that installs the runtime, downloads the model and launches SGLang.

## In this article
- Why Terraform could help for SGLang deployments.
- Install the Verda Terraform provider (custom provider).
- Terraform configuration: Singlenode instance 4xB300 with SSH key, local NVMe volume, and startup script.
- Startup script: Local NVMe setup, Docker configuration, model download and SGLang server launch.

## SGLang

As introduced in previous blogs. We further continue research effort on the SGLang serving framework. Which up to this date covers not only model inference but Reinforcement Learning (RL) post-training techniques integrating with [Miles](https://github.com/radixark/miles), [Slime](https://github.com/THUDM/slime) and [veRL](https://github.com/volcengine/verl).

Their work through 2025 focused on leveraging the optimizations performed and presented by the DeepSeek team in *DeepSeek-V3 Technical Report.* Being one of these Prefill-Decode disaggregation (PD disaggregation) one of the most complex to orchestrate resource-wise. Thus, Infrastructure as a Code (IaC) tools like Terraform play a significant role in managing this complexity.

## Terraform

Terraform is an open-source Infrastructure as Code (IaC) tool by HashiCorp that enables defining, provisioning, and managing cloud and on-premises infrastructure using their declarative language (HCL) in configuration files, thus ensuring versioning, reuse, and automation for safe, efficient infrastructure deployment across cloud platforms.

This enables systematic deployment of resources to scale flexibly, while ensuring resources are trackable and identified through the whole system.

### Basics of Terraform


The core Terraform concepts are:

* **providers**: plugins to interact with cloud providers APIs.  
* **resources**: which Terraform handles via providers.  
* **data sources**: read-only lookups.  
* **modules**: reusable bundles of Terraform configs.  
* **variables / outputs**: parameters in, results out.

In most common cases, we will be running combinations of these commands:

```bash
terraform init: sets up the working directory, downloads providers/modules.
terraform plan: dry run/safe mode which tells what would change based on current state.
terraform apply: performs the proposed changes.
terraform destroy: deletes resources.
```

By default, Terraform saves the current state of infrastructure and resources on a local file called terraform.tfstate. We can consult the list of resources deployed using `terraform state list`.

### Setup

Since the Verda provider isn’t yet on the Terraform Registry, you’ll install it locally:

1. Download and install Terraform:

```bash
# Terraform installation
wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(grep -oP '(?<=UBUNTU_CODENAME=).*' /etc/os-release || lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

2. Download and install Go:

```bash
# Go installation: https://go.dev/dl/
wget https://go.dev/dl/go1.25.6.linux-386.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.25.6.linux-386.tar.gz
export PATH=$PATH:/usr/local/go/bin

# verification
go version
```

3. Download the terraform-verda-provider from our [Github repository](https://github.com/verda-cloud/terraform-provider-verda/tree/main):

```bash
git clone https://github.com/verda-cloud/terraform-provider-verda.git
```

4. Build the verda provider using Go:

```bash
cd terraform-provider-verda
go build -o terraform-provider-verda

```

5. Once built, we specify the location of the binary on the `.terraformrc` CLI config file, this file resides on the home folder of the user. Written in HCL too, It enables custom configurations as `provider_installation`, which customizes the installation methods used by `terraform init` when installing provider plugins.

```bash
# Terraform custom provider 
# NOTE: change verda-cloud/verda value with the correct path

echo 'provider_installation {
 dev_overrides {
   "verda-cloud/verda" = "<Path_to_git_clone_verda_repo>/terraform-provider-verda"
 }
 direct {}
}' > $HOME/.terraformrc
```

### Terraform configuration

Once we finish the setup we can create our terraform configurations for managing resources. Terraform configurations are `.tf` files which describe resources for the infrastructure we will deploy, specifying their properties and the dependencies between them.

This configuration describes several resources:

* `terraform` block: Parent block that contains configurations that define Terraform behavior.   
* `required_providers` block: Specifies all provider plugins required to create and manage resources specified in the configuration.  
* `resource` block: Dependent on the provider configuration for their resources. In our case resources are specified [here](https://github.com/verda-cloud/terraform-provider-verda?tab=readme-ov-file\#resources).  
* `output` block: Exposes information about your infrastructure that you can reference on the command line or any other Terraform configurations, enabling configurations of programmatic capabilities.


```hcl
# NOTE: Path to public key required on public_key

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
  public_key = file("<path_to_key.pub>")
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
  os_volume = {
    name = "terraform-os"
    size = 200
    type = "NVMe"
  }

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
```

In order to define the properties, we will need to consult the Verda Cloud API. For this, we firstly require Cloud API Credentials which can be generated in the dashboard (docs [here](https://api.datacrunch.io/v1/docs\#description/verda-cloud)). 

Secondly, we need an access token for requesting the API which can be obtained following the previous steps or by consulting the request documentation [here](https://api.datacrunch.io/v1/docs\#tag/authentication/POST/v1/oauth2/token).

Our provider supports the following methods for providing credentials:

* **Provider Configuration** (shown above on `provider` block)  
* **Environment Variables**:  
  * `VERDA_CLIENT_ID`  
  * `VERDA_CLIENT_SECRET`  
  * `VERDA_BASE_URL` (optional)

We provide the following snippet for requesting relevant information to the Verda API:

```bash
#! /bin/bash

BASE_URL="${BASE_URL:-https://api.datacrunch.io/v1}"
VERDA_CLIENT_ID=$VERDA_CLIENT_ID
VERDA_CLIENT_SECRET=$VERDA_CLIENT_SECRET

RESP=$(
 curl -sS --request POST "$BASE_URL/oauth2/token" \
   --header "Content-Type: application/json" \
   --data "{\"grant_type\":\"client_credentials\",\"client_id\":\"$VERDA_CLIENT_ID\",\"client_secret\":\"$VERDA_CLIENT_SECRET\"}"
)

#echo "Raw response:"
echo "$RESP"

if command -v jq >/dev/null 2>&1; then
 TOKEN="$(echo "$RESP" | jq -r '.access_token')"
 echo
 echo "access_token:"
 echo "$TOKEN"
fi

echo
RESP=$(
 curl -sS --request GET "$BASE_URL/instance-types" \
     --header "Authorization: Bearer $TOKEN" \
   )

echo "instances types:"
echo "$RESP" | jq -r '.[].instance_type'

echo
RESP=$(
 curl -sS --request GET "$BASE_URL/images" \
     --header "Authorization: Bearer $TOKEN" \
   )

echo "images:"
echo "$RESP" | jq -r '.[].image_type'


echo
```

Once we have specified the resources we want to deploy, we can test how it will affect our current terraform state by executing `terraform plan`. And apply the subsequent changes with `terraform apply`.

> **Tip:** we can consult the IP and the status of the instance created by doing terraform plan as we created both outputs: `instance_ip` and `instance_status`.

## Model Deployment

To deploy, for this example, Deepseek R1 NVFP4-quantized, we include the following script which will be executed once the instance is deployed. As we can see above this script is included as a resource `resource "verda_startup_script" "init_vm" {...}`.

> A Huggingface Token is required to download the model from their hub.

```bash
#!/bin/bash

export HOST_MODEL_PATH="/mnt/local_nvme/models"
export C_MODEL_PATH="/root/models"
export HF_TOKEN=$HF_TOKEN

# Create a local NVMe volume
mkdir /mnt/local_nvme
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%
sudo partprobe /dev/vdb
sudo mkfs.ext4 -F /dev/vdb1
sudo mount /dev/vdb1 /mnt/local_nvme

mkdir -p $HOST_MODEL_PATH


# Docker setup (store artifacts in local NVMe)
sudo mkdir -p /mnt/local_nvme/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
   "data-root": "/mnt/local_nvme/docker",
   "runtimes": {
       "nvidia": {
           "args": [],
           "path": "nvidia-container-runtime"
       }
   }
}
EOF


# Restart docker to apply the changes
sudo systemctl restart docker

# sglang setup
# check image version in https://hub.docker.com/r/lmsysorg/sglang/tags
docker run --gpus all --shm-size 32g --network=host --name sglang_server -d --ipc=host \
 -v "$HOST_MODEL_PATH:$C_MODEL_PATH" \
 -e HF_TOKEN="$HF_TOKEN" \
 lmsysorg/sglang:dev-cu13 \
 bash -lc "
   huggingface-cli download nvidia/DeepSeek-R1-0528-NVFP4-v2 --cache-dir "$C_MODEL_PATH"
   exec python3 -m sglang.launch_server \
     --model-path "$C_MODEL_PATH/models--nvidia--DeepSeek-R1-0528-NVFP4-v2/snapshots/25a138f28f49022958b9f2d205f9b7de0cdb6e18/" \
     --served-model-name dsr1 \
     --tp 4 \
     --attention-backend trtllm_mla \
     --disable-radix-cache \
     --moe-runner-backend flashinfer_trtllm \
     --quantization modelopt_fp4 \
     --kv-cache-dtype fp8_e4m3
 "

# Check logs: docker logs -f sglang_server
# Execute benchmark with: docker exec -it sglang_server bash -lc "python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 1024"
```

### Benchmarks

As a quick test we include the results of executing the following SGLang benchmark on the deployed instance:

```bash
docker exec -it sglang_server bash -lc "python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 1024"
```

```bash
root@terraform-sglang:~# docker exec -it sglang_server bash -lc "python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 1024"
======== Warmup Begin ========
Warmup with batch_size=[16]
#Input tokens: 16384
#Output tokens: 256
batch size: 16
input_len: 1024
output_len: 16
latency: 0.50 s
input throughput: 50249.39 tok/s
output throughput: 1477.75 tok/s
last_ttft: 0.33 s
last generation throughput: 1337.26 tok/s
======== Warmup End   ========

#Input tokens: 16384
#Output tokens: 16384
batch size: 16
input_len: 1024
output_len: 1024
latency: 12.55 s
input throughput: 50307.77 tok/s
output throughput: 1340.45 tok/s
last_ttft: 0.33 s
last generation throughput: 1337.14 tok/s

Results are saved to result.jsonl
```

This makes the deployment and benchmark reproducible end-to-end.

## Future work

This tutorial covers a single-node deployment. Next, we propose to exetend it to multi-component serving setups where IaC really excel:

- **PD disaggregation on Kubernetes**: use Terraform to provision a K8s cluster and separate GPU node pools for prefill and decode, then deploy SGLang PD disaggregated mode to orchestrate it.
- **Autoscaling policies**: scale decode replicas on queue depth / tokens-per-second, and scale prefill on bursty traffic.
- **Benchmark automation**: run reproducible PD benchmarks as K8s Jobs and store results and configs for apples-to-apples comparisons.

# References

- [Run DeepSeek-R1 on AWS EC2 Using Ollama](https://www.pulumi.com/blog/run-deepseek-on-aws-ec2-using-pulumi/).  

- [SGlang slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file\#slides).  

- [Verda Terraform provider](https://github.com/verda-cloud/terraform-provider-verda).  

- [Terraform Provider Registry](https://registry.terraform.io/browse/providers).  

- [Terraform documentation](https://developer.hashicorp.com/terraform).  

- [Terraform configuration files](https://developer.hashicorp.com/terraform/language).  

- [Terraform resources](https://developer.hashicorp.com/terraform/language/block/resource).  

- [SGLang benchmark and profiling](https://docs.sglang.io/developer\_guide/benchmark\_and\_profiling.html).  
