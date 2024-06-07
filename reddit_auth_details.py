class RedditUser():
    def __init__(self,id):
        self.id = id
        self.client_id = None
        self.client_secret = None
        self.user_agent = None
        self.username = None
        self.password = None


    def LoadUser(self, id):
        if id == 1:
            self.client_id = 'Ynu_EYMg7FJKJc9m-0FbUg'
            self.client_secret = 'vqbs5rwZmyJ0DhRpOUAAFruRxkyKTQ'
            self.user_agent = 'oruni/0.1 by u/pro_skraper'
            self.username = 'pro_skraper'
            self.password = 'Thisis2021!R'
        elif id == 2:
            self.client_id = 'mHmUQZOkDm-9Npe6NC00Dw'
            self.client_secret = 'qI25uK1WT3XXTq8j8jvCvys3DC82vg'
            self.user_agent = 'for personal usage and Academic course on NLP and SNA'
            self.username = 'Logan_XDA'
            self.password = '2P.a6bf-@v$R!P'

