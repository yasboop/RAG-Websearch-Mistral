"""
Script to deploy the Gromo RAG Chatbot to AWS EC2.
"""
import os
import argparse
import boto3
import paramiko
import time
from botocore.exceptions import ClientError

from src.config import AWS_REGION, AWS_INSTANCE_TYPE

# AWS EC2 settings
AMI_ID = "ami-0c7217cdde317cfec"  # Amazon Linux 2023 AMI (x86_64)
KEY_NAME = "gromo-rag-key"  # Name of the key pair to use
SECURITY_GROUP_NAME = "gromo-rag-sg"  # Name of the security group to use
INSTANCE_NAME = "gromo-rag-instance"  # Name of the EC2 instance

# Setup script to run on the EC2 instance
SETUP_SCRIPT = """#!/bin/bash
# Update system packages
sudo yum update -y

# Install Python and development tools
sudo yum install -y python3 python3-pip python3-devel git gcc

# Install NVIDIA drivers and CUDA (for GPU instances)
sudo yum install -y gcc kernel-devel-$(uname -r)
sudo amazon-linux-extras install -y epel
sudo yum install -y dkms
sudo yum install -y nvidia cuda

# Clone the repository
git clone https://github.com/yourusername/gromo-rag.git
cd gromo-rag

# Install dependencies
pip3 install -r requirements.txt
pip3 install bitsandbytes

# Set up environment variables
echo "OPENAI_API_KEY=your_openai_api_key" > .env

# Initialize the vector store
python3 init_vector_store.py

# Install and configure Nginx
sudo yum install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Create a systemd service for the FastAPI app
cat << EOF | sudo tee /etc/systemd/system/gromo-rag.service
[Unit]
Description=Gromo RAG Chatbot
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/gromo-rag
ExecStart=/usr/bin/python3 app_fastapi.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=gromo-rag

[Install]
WantedBy=multi-user.target
EOF

# Start and enable the service
sudo systemctl start gromo-rag
sudo systemctl enable gromo-rag

# Configure Nginx as a reverse proxy
cat << EOF | sudo tee /etc/nginx/conf.d/gromo-rag.conf
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Reload Nginx configuration
sudo systemctl reload nginx

echo "Deployment completed successfully!"
"""

def create_key_pair(ec2_client, key_name):
    """
    Create a key pair for SSH access to the EC2 instance.
    
    Args:
        ec2_client: The EC2 client
        key_name (str): The name of the key pair
        
    Returns:
        str: The path to the private key file
    """
    try:
        # Create key pair
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
        
        # Save private key to file
        private_key_path = f"{key_name}.pem"
        with open(private_key_path, "w") as f:
            f.write(key_pair["KeyMaterial"])
        
        # Set permissions on private key file
        os.chmod(private_key_path, 0o400)
        
        print(f"Created key pair: {key_name}")
        return private_key_path
    
    except ClientError as e:
        if "already exists" in str(e):
            print(f"Key pair {key_name} already exists")
            return f"{key_name}.pem"
        else:
            raise

def create_security_group(ec2_client, security_group_name):
    """
    Create a security group for the EC2 instance.
    
    Args:
        ec2_client: The EC2 client
        security_group_name (str): The name of the security group
        
    Returns:
        str: The ID of the security group
    """
    try:
        # Create security group
        security_group = ec2_client.create_security_group(
            GroupName=security_group_name,
            Description="Security group for Gromo RAG Chatbot"
        )
        security_group_id = security_group["GroupId"]
        
        # Add inbound rules
        ec2_client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                # SSH
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
                },
                # HTTP
                {
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
                },
                # HTTPS
                {
                    "IpProtocol": "tcp",
                    "FromPort": 443,
                    "ToPort": 443,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
                }
            ]
        )
        
        print(f"Created security group: {security_group_name}")
        return security_group_id
    
    except ClientError as e:
        if "already exists" in str(e):
            # Get existing security group ID
            response = ec2_client.describe_security_groups(
                GroupNames=[security_group_name]
            )
            security_group_id = response["SecurityGroups"][0]["GroupId"]
            print(f"Security group {security_group_name} already exists")
            return security_group_id
        else:
            raise

def launch_ec2_instance(ec2_client, ami_id, instance_type, key_name, security_group_id, instance_name):
    """
    Launch an EC2 instance.
    
    Args:
        ec2_client: The EC2 client
        ami_id (str): The ID of the AMI to use
        instance_type (str): The type of instance to launch
        key_name (str): The name of the key pair to use
        security_group_id (str): The ID of the security group to use
        instance_name (str): The name of the instance
        
    Returns:
        str: The ID of the instance
    """
    # Launch instance
    response = ec2_client.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "Name",
                        "Value": instance_name
                    }
                ]
            }
        ]
    )
    
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"Launched instance: {instance_id}")
    
    # Wait for instance to be running
    print("Waiting for instance to be running...")
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    
    # Get instance details
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    public_ip = response["Reservations"][0]["Instances"][0]["PublicIpAddress"]
    
    print(f"Instance is running at: {public_ip}")
    return instance_id, public_ip

def wait_for_ssh(public_ip, private_key_path, username="ec2-user", timeout=300):
    """
    Wait for SSH to be available on the instance.
    
    Args:
        public_ip (str): The public IP of the instance
        private_key_path (str): The path to the private key file
        username (str): The username to use for SSH
        timeout (int): The timeout in seconds
        
    Returns:
        bool: True if SSH is available, False otherwise
    """
    print(f"Waiting for SSH to be available at {public_ip}...")
    
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Try to connect until timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            ssh.connect(
                public_ip,
                username=username,
                key_filename=private_key_path,
                timeout=5
            )
            print("SSH is available!")
            ssh.close()
            return True
        except Exception as e:
            print(f"SSH not yet available: {e}")
            time.sleep(10)
    
    print(f"Timed out waiting for SSH after {timeout} seconds")
    return False

def deploy_application(public_ip, private_key_path, setup_script, username="ec2-user"):
    """
    Deploy the application to the EC2 instance.
    
    Args:
        public_ip (str): The public IP of the instance
        private_key_path (str): The path to the private key file
        setup_script (str): The setup script to run on the instance
        username (str): The username to use for SSH
    """
    print(f"Deploying application to {public_ip}...")
    
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to the instance
        ssh.connect(
            public_ip,
            username=username,
            key_filename=private_key_path
        )
        
        # Create setup script on the instance
        setup_script_path = "/tmp/setup.sh"
        stdin, stdout, stderr = ssh.exec_command(f"cat > {setup_script_path}")
        stdin.write(setup_script)
        stdin.flush()
        stdin.channel.shutdown_write()
        
        # Make setup script executable
        ssh.exec_command(f"chmod +x {setup_script_path}")
        
        # Run setup script
        print("Running setup script...")
        stdin, stdout, stderr = ssh.exec_command(f"sudo {setup_script_path}")
        
        # Print output
        for line in stdout:
            print(line.strip())
        
        # Print errors
        for line in stderr:
            print(f"ERROR: {line.strip()}")
        
        print("Deployment completed!")
    
    except Exception as e:
        print(f"Error deploying application: {e}")
    
    finally:
        ssh.close()

def main():
    """
    Main function to deploy the Gromo RAG Chatbot to AWS EC2.
    """
    parser = argparse.ArgumentParser(description="Deploy Gromo RAG Chatbot to AWS EC2")
    parser.add_argument("--region", default=AWS_REGION, help="AWS region")
    parser.add_argument("--instance-type", default=AWS_INSTANCE_TYPE, help="EC2 instance type")
    parser.add_argument("--ami-id", default=AMI_ID, help="AMI ID")
    parser.add_argument("--key-name", default=KEY_NAME, help="Key pair name")
    parser.add_argument("--security-group-name", default=SECURITY_GROUP_NAME, help="Security group name")
    parser.add_argument("--instance-name", default=INSTANCE_NAME, help="Instance name")
    args = parser.parse_args()
    
    # Create EC2 client
    ec2_client = boto3.client("ec2", region_name=args.region)
    
    try:
        # Create key pair
        private_key_path = create_key_pair(ec2_client, args.key_name)
        
        # Create security group
        security_group_id = create_security_group(ec2_client, args.security_group_name)
        
        # Launch EC2 instance
        instance_id, public_ip = launch_ec2_instance(
            ec2_client,
            args.ami_id,
            args.instance_type,
            args.key_name,
            security_group_id,
            args.instance_name
        )
        
        # Wait for SSH to be available
        if wait_for_ssh(public_ip, private_key_path):
            # Deploy application
            deploy_application(public_ip, private_key_path, SETUP_SCRIPT)
            
            print(f"\nDeployment successful!")
            print(f"You can access the chatbot at: http://{public_ip}")
            print(f"SSH access: ssh -i {private_key_path} ec2-user@{public_ip}")
        else:
            print("Failed to connect to the instance via SSH")
    
    except Exception as e:
        print(f"Error deploying to AWS EC2: {e}")

if __name__ == "__main__":
    main() 