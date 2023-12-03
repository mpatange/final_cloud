provider "aws" {
  region = "us-west-2"  # Set the AWS region
}

data "aws_ami" "latest_amazon_linux" {
  most_recent = true

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]  # Filter for the Amazon Linux AMI
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]  # Specify the virtualization type
  }

  owners = ["amazon"]  # Specify the owner of the AMI
}

resource "aws_instance" "example" {
  ami           = data.aws_ami.latest_amazon_linux.id  # Use the latest Amazon Linux AMI
  instance_type = "t2.micro"  # Specify the instance type

  tags = {
    Name = "mis547_terraform"  # Tag the instance
  }
}
