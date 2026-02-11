# Verda Terraform provider
## Quickstart
1. Build the provider binary

```bash
cd terraform-provider-verda
go build -o terraform-provider-verda
```

2. Point to the local built provider

As Verda provider isn’t in the Terraform Registry yet, we  use a dev override pointing Terraform to the locally built provider binary.

```bash
# in $HOME/.terraformrc
provider_installation {
  dev_overrides {
    "verda-cloud/verda" = "/mnt/cephfs/dc/rodri/IaC_blog/terraform-provider-verda/terraform-provider-verda"
  }
  direct {}
}
```

## Resources:
- `verda_instance`: create a compute instance
- `verda_volume`: create a storage volume
- `verda_ssh_key`: upload an SSH key
- `verda_startup_script`: startup scripts for instances
- `verda_containe`r: a serverless container deployment (with scaling)
- `verda_serverless_jo`b: a serverless “job” deployment
- `verda_container_registry_credentials`: auth for private registries
