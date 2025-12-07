from models import Session, User

def authenticate(username, password, role):
    session = Session()
    user = session.query(User).filter_by(username=username, password=password, role=role).first()
    session.close()
    return user