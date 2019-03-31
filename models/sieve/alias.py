import requests
import urllib.parse


# Class to find aliases for named entities from wikipedia
class Alias:

    # ULR to work with API
    url_api = "https://uk.wikipedia.org/w/api.php"

    # HTTP client to send requests
    http_client = None

    # Limitation of Wikipedia to info request
    wikipedia_max_count = 40

    # def __init__(self):
    #     self.http_client = http.client.HTTPSConnection(self.url_api)
    #     self.http_client.set_debuglevel(2)

    # Split list into chunks
    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # Find wikipedia links for each named entity
    def find_wikipedia_links(self, entities):
        search_titles = {}

        entity_links = {}

        # Loop through entities array
        for entity_id in entities:
            entity = entities[entity_id]
            words = []

            # Check if the entity is proper name
            is_proper_name = True
            for token in entity:
                if not token.IsProperName:
                    is_proper_name = False
                    break
                # Append word to the list
                words.append(token.Lemmatized)
            if is_proper_name:
                search_titles[entity_id] = ' '.join(words)
        # If there is more named entities than 2
        if len(search_titles) > 1:
            # Get keys of dictionary and split them into several parts to keep up wikipedia requirements
            keys = list(search_titles.keys())
            chunks = list(self.chunks(keys, 40))

            # Loop through chunks and send request
            for chunk in chunks:
                # Form title
                words_chunk = []
                for entity_id in chunk:
                    words_chunk.append(search_titles[entity_id])
                title = "|".join(words_chunk)
                params = {
                    'action': 'query',
                    'format': 'json',
                    'prop': 'redirects',
                    'titles': title
                }
                print(title)
                response = requests.post(self.url_api, params)
                if response.status_code == 200:
                    data = response.json()
                    if "query" in data and "pages" in data["query"]:
                        pages = data["query"]["pages"]
                        counter = 0
                        for key in pages:
                            print(pages[key]["title"])
                            page_id = int(key)
                            if page_id > 0:
                                entity_id = chunk[counter]
                                entity_links[entity_id] = pages[key]
                            counter += 1
        return entity_links

