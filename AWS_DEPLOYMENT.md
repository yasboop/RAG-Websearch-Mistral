# Deploying Gromo RAG Chatbot to AWS

This guide explains how to deploy the Gromo RAG Chatbot to AWS using the provided deployment script.

## Prerequisites

Before deploying to AWS, make sure you have:

1. An AWS account with free tier credits
2. AWS CLI installed and configured with your credentials
3. Python 3.8 or higher installed
4. Required Python packages installed:
   ```
   pip install boto3 paramiko
   ```

## Deployment Steps

### 1. Prepare Your Code

Make sure your code is ready for deployment:

- All required files are in place
- The vector store is initialized locally (optional)
- Configuration settings are correct in `src/config.py`

### 2. Configure AWS Credentials

If you haven't already, configure your AWS credentials:

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region, and output format.

### 3. Run the Deployment Script

Run the deployment script:

```bash
python deploy_aws_ec2.py
```

This script will:

1. Create a key pair for SSH access
2. Create a security group with necessary inbound rules
3. Launch an EC2 instance with the specified AMI and instance type
4. Wait for the instance to be running and SSH to be available
5. Deploy the application to the instance
6. Configure Nginx as a reverse proxy
7. Start the application as a systemd service

### 4. Access the Chatbot

Once the deployment is complete, you can access the chatbot at:

```
http://<instance-public-ip>
```

The deployment script will output the URL and SSH access command.

### 5. Monitoring and Management

To monitor the application logs:

```bash
ssh -i <key-pair>.pem ec2-user@<instance-public-ip>
sudo journalctl -u gromo-rag
```

To restart the application:

```bash
sudo systemctl restart gromo-rag
```

## Using AWS Free Tier

The deployment is configured to use a g4dn.xlarge instance, which is not part of the free tier but can be covered by AWS free credits. The instance is optimized for running the Qwen 7B model with quantization.

If you want to use only free tier resources, modify the `AWS_INSTANCE_TYPE` in `src/config.py` to `t2.micro` or `t3.micro`, but note that you'll need to use a much smaller model or an external API.

## Cleaning Up

To avoid incurring charges, remember to terminate your EC2 instance when you're done:

```bash
aws ec2 terminate-instances --instance-ids <instance-id>
```

You can also delete the security group and key pair:

```bash
aws ec2 delete-security-group --group-name gromo-rag-sg
aws ec2 delete-key-pair --key-name gromo-rag-key
```

## Troubleshooting

If you encounter issues during deployment:

1. Check the deployment script output for errors
2. Verify that your AWS credentials are correct
3. Ensure that you have sufficient permissions to create EC2 instances
4. Check the instance logs for application errors
5. Make sure the security group allows inbound traffic on port 80

For application-specific issues, SSH into the instance and check the logs:

```bash
sudo journalctl -u gromo-rag
``` 