#
# import os
# from channels.routing import ProtocolTypeRouter, URLRouter
# from django.core.asgi import get_asgi_application
# from channels.auth import AuthMiddlewareStack
# # import chat.routing
# # import video.routing
# from . import routing
#
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mydjango.settings')
#
# application = ProtocolTypeRouter({
#     "http": get_asgi_application(),
#     "websocket": AuthMiddlewareStack(
#     URLRouter(
#     routing.websocket_urlpatterns
#     )
#     ),
# })

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack

from mydjango import routing



os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mydjango.settings')

# application = get_asgi_application()
# application = ProtocolTypeRouter({
#     'http': get_asgi_application(),
#     'websocket': URLRouter(routing.websocket_urlpatterns),
# })
application = ProtocolTypeRouter({
  "http": get_asgi_application(),
  "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns
        )
    ),
})
