import webbrowser

from flask import Flask, redirect, url_for, session, request
from flask_oauth import OAuth, OAuthException


SECRET_KEY = 'development key'
DEBUG = True
FACEBOOK_APP_ID = '865933310126748'
FACEBOOK_APP_SECRET = 'f9b5c2a93f2481dd737da8ef02bee6e9'


app = Flask(__name__)
app.debug = DEBUG
app.secret_key = SECRET_KEY
oauth = OAuth()

facebook = oauth.remote_app('facebook',
    base_url='https://graph.facebook.com/',
    request_token_url=None,
    access_token_url='/oauth/access_token',
    authorize_url='https://www.facebook.com/dialog/oauth',
    consumer_key=FACEBOOK_APP_ID,
    consumer_secret=FACEBOOK_APP_SECRET,
    request_token_params={'scope': 'email, user_photos'}
)


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login')
def login():
    return facebook.authorize(callback=url_for('facebook_authorized',
        next=request.args.get('next') or request.referrer or None,
        _external=True))


@app.route('/login/authorized')
@facebook.authorized_handler
def facebook_authorized(resp):
    if resp is None:
        return 'Access denied: reason=%s error=%s' % (
            request.args['error_reason'],
            request.args['error_description']
        )
    session['oauth_token'] = (resp['access_token'], '')
    me = facebook.get('/me')
    return 'Logged in as id=%s<br/>name=%s<br/>redirect=%s<br/>access_token=<pre>%s</pre>' % \
        (me.data['id'], me.data['name'], request.args.get('next'), session['oauth_token'])


@facebook.tokengetter
def get_facebook_oauth_token():
    return session.get('oauth_token')


if __name__ == '__main__':
    url = "http://www.cs596-facebook-auth.com:5000"
    webbrowser.open(url, new=1)
    app.run()
