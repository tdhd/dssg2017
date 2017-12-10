"""
WSGI config for dkg project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

from .settings import USER_NAME, PASSWORD


class AuthenticationMiddleware(object):
    def __init__(self, app, username, password):
        self.app = app
        self.username = username
        self.password = password

    def __unauthorized(self, start_response):
        start_response('401 Unauthorized', [
            ('Content-type', 'text/plain'),
            ('WWW-Authenticate', 'Basic realm="restricted"')
        ])
        return ['You are unauthorized and forbidden to view this resource.']

    def __call__(self, environ, start_response):
        authorization = environ.get('HTTP_AUTHORIZATION', None)
        if not authorization:
            return self.__unauthorized(start_response)

        (method, authentication) = authorization.split(' ', 1)
        if 'basic' != method.lower():
            return self.__unauthorized(start_response)

        request_username, request_password = authentication.strip().decode('base64').split(':', 1)
        if self.username == request_username and self.password == request_password:
            return self.app(environ, start_response)

        return self.__unauthorized(start_response)


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dkg.settings")

application = get_wsgi_application()
application = AuthenticationMiddleware(application, USER_NAME, PASSWORD)
