provider "aws" {
  region = "us-west-2"  # Replace with your desired AWS region
}

data "aws_ami" "latest_amazon_linux" {
  most_recent = true

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["amazon"]
  region = "us-west-2"
}

#Dynamicly take latest ami id of us-west-2

resource "aws_instance" "example" {
  ami           = data.aws_ami.latest_amazon_linux.id
  instance_type = "t2.micro"
  tags = {
    Name = "mis547_terraform"
  }
}


