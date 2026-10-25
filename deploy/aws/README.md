# AWS EC2 deployment

`cloudformation.yaml` runs the existing Streamlit app on Amazon EC2. It is not an Amplify build specification.

## Resources to review before launch

- One `t3.micro` (1 GiB RAM), with an optional `t3.small` setting.
- One encrypted 8 GiB gp3 root disk, removed on instance termination.
- One automatically assigned public IPv4 address.
- A dedicated VPC, public subnet, internet gateway, route table, and security group.
- Public inbound HTTP on TCP 80. No IAM role, instance profile, inbound SSH, or public Streamlit port is created.
- Standard CPU credits; no T3 Unlimited surplus-credit charging.

EC2, EBS, public IPv4, and applicable data transfer normally incur charges. On the AWS Free account plan they consume credits, and the account closes when credits or the plan period expire. A paid account can receive bills after credits expire. For the user's zero-charge requirement, launch only after verifying the console explicitly says the account is on the Free plan, and never upgrade it. Budget alerts alone do not enforce a zero-charge limit.

## What starts automatically

Amazon Linux 2023 installs Python 3.12 and the project's pinned requirements. The template checks out the specified Git commit, verifies the saved model, and starts Streamlit as an unprivileged systemd service behind Nginx. Nginx forwards WebSocket upgrades. The service restarts after failure and starts on boot.

The model is reused; deployment does not train it. The default revision is `3cc7df8111a4232c86377783b57705504270cdb1`. Changing a parameter on an existing instance does not rerun user data automatically; review a replacement/update plan before upgrading a running deployment.

CloudFormation waits up to 20 minutes for the bootstrap health signal. A failed bootstrap marks creation unsuccessful. Bootstrap output is recorded in `/var/log/lung-app-bootstrap.log` and the instance console. CloudFormation rollback may remove newly created resources.

## Console launch

1. Open CloudFormation in the chosen AWS region and select **Create stack → With new resources**.
2. Upload `cloudformation.yaml` and choose a stack name, such as `lung-numerical-demo`.
3. Verify the account is still on the Free plan; review the instance size, code revision, credit consumption, and networking.
4. Submit only when zero-charge Free plan coverage is verified. This template does not require an IAM capability acknowledgement. Stop if AWS requires upgrading the account to deploy it.
5. Wait for `CREATE_COMPLETE`, then open `ApplicationUrl` in Outputs.
6. Verify Home, Evaluation, numerical prediction, and CSV prediction in the deployed app.

The initial URL uses HTTP. Use demonstration inputs until a custom domain and HTTPS are configured. The automatically assigned IP can change after stop/start. There is no load balancer, NAT gateway, Elastic IP, custom domain, or automatic GitHub deployment in this template.

## Operations

Bootstrap output is available in EC2 → Actions → Monitor and troubleshoot → Get system log. This deployment creates no SSH key, IAM administration role, or Session Manager access. Review a replacement plan for code updates and recovery. Stopping an instance stops its compute usage, but EBS storage and other retained resources still consume credits or incur charges on paid accounts. Review stack deletion before cleanup; deletion removes this server and its root disk.

## Validation status

Template/schema and bootstrap syntax checks are performed before launch. A successful live AWS deployment and public prediction check must still be recorded after launch; local validation alone does not establish that the cloud installation succeeds.
