# AWS EC2 deployment

`cloudformation.yaml` runs the existing Streamlit app on Amazon EC2. It is not an Amplify build specification.

## Resources to review before launch

- One `t3.micro` (1 GiB RAM), with an optional `t3.small` setting.
- One encrypted 8 GiB gp3 root disk, removed on instance termination.
- One automatically assigned public IPv4 address.
- A dedicated VPC, public subnet, internet gateway, route table, and security group.
- An EC2 IAM role and instance profile with `AmazonSSMManagedInstanceCore` for administration through Systems Manager.
- Public inbound HTTP on TCP 80. No inbound SSH or Streamlit port is opened.
- Standard CPU credits; no T3 Unlimited surplus-credit charging.

EC2, EBS, public IPv4, and applicable data transfer can incur charges. Free Tier or credits depend on the account. Creating a stack requires explicit approval for these resources and IAM permissions.

## What starts automatically

Amazon Linux 2023 installs Python 3.12 and the project's pinned requirements. The template checks out the specified Git commit, verifies the saved model, and starts Streamlit as an unprivileged systemd service behind Nginx. Nginx forwards WebSocket upgrades. The service restarts after failure and starts on boot.

The model is reused; deployment does not train it. The default revision is `3cc7df8111a4232c86377783b57705504270cdb1`. Changing a parameter on an existing instance does not rerun user data automatically; review a replacement/update plan before upgrading a running deployment.

CloudFormation waits up to 20 minutes for the bootstrap health signal. A failed bootstrap marks creation unsuccessful. Bootstrap output is recorded in `/var/log/lung-app-bootstrap.log` and the instance console. CloudFormation rollback may remove newly created resources.

## Console launch

1. Open CloudFormation in the chosen AWS region and select **Create stack → With new resources**.
2. Upload `cloudformation.yaml` and choose a stack name, such as `lung-numerical-demo`.
3. Review the instance size, code revision, costs, networking, and IAM role.
4. Acknowledge IAM resource creation only after approving the role. Submit the stack after approving the AWS charges.
5. Wait for `CREATE_COMPLETE`, then open `ApplicationUrl` in Outputs.
6. Verify Home, Evaluation, numerical prediction, and CSV prediction in the deployed app.

The initial URL uses HTTP. Use demonstration inputs until a custom domain and HTTPS are configured. The automatically assigned IP can change after stop/start. There is no load balancer, NAT gateway, Elastic IP, custom domain, or automatic GitHub deployment in this template.

## Operations

Use EC2 → Connect → Session Manager for administration after the agent registers. Service logs are available with `journalctl -u lung-app`. Stopping an instance stops its compute billing, but EBS storage and other retained resources can still incur charges. Review stack deletion before cleanup; deletion removes this server and its root disk.

## Validation status

Template/schema and bootstrap syntax checks are performed before launch. A successful live AWS deployment and public prediction check must still be recorded after launch; local validation alone does not establish that the cloud installation succeeds.
