from clients.clientBase import ClientBase


class ClientFedAvg(ClientBase):
    def __init__(self, args, client_id, data):
        super(ClientFedAvg, self).__init__(args, client_id, data)
