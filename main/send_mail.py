import smtplib
from email.mime.text import MIMEText
from email.header import Header
import sys, os
from dotenv import load_dotenv
load_dotenv()

def send_email(subject, content):
    # --- 配置信息 ---
    sender =  os.getenv("email")          
    password = os.getenv("password")     
    receiver = os.getenv("email")    
    smtp_server = 'smtp.qq.com'

    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = sender
    message['To'] = receiver
    message['Subject'] = Header(subject, 'utf-8')

    try:
        # QQ邮箱使用 SSL 端口 465
        smtp_obj = smtplib.SMTP_SSL(smtp_server, 465)
        smtp_obj.login(sender, password)
        smtp_obj.sendmail(sender, [receiver], message.as_string())
        smtp_obj.quit()
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        send_email(sys.argv[1], sys.argv[2])