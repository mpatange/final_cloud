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

resource "aws_security_group" "allow_web" {
  name        = "allow_web_traffic"
  description = "Allow web inbound traffic"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "django-app" {
  ami           = data.aws_ami.latest_amazon_linux.id
  instance_type = "t2.micro"
  security_groups = [aws_security_group.allow_web.name]

  tags = {
    Name = "mis547_terraform"
  }
  
  user_data = <<-EOF
                #!/bin/bash
                sudo yum update -y
                sudo yum install -y docker
                sudo service docker start
                sudo docker pull mpatange/cloudproject:django-app
                sudo docker run -d -p 80:8000 mpatange/cloudproject:django-app
                EOF

  connection {
    type        = "ssh"
    user        = "ec2-user"
    private_key = file("${path.module}/cloud-key.pem")
    host        = self.public_ip
  }
}

output "public_ip" {
  value = aws_instance.django-app.public_ip
}
