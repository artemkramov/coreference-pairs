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

        # Resulting dictionary that contains entity ID with redirect links
        entity_links = {}

        # Dictionary that will group all entities with the same lemmatized version
        title_groups = {}

        # Loop through entities array
        for entity_id in entities:
            entity = entities[entity_id]
            words_lemma = []
            words_raw = []

            # Check if the entity is proper name
            is_proper_name = True
            for token in entity:
                if not token.IsProperName:
                    is_proper_name = False
                    break
                # Append word to the list
                words_lemma.append(token.Lemmatized)
                words_raw.append(token.RawText)

            # Add just named entities
            # Group all entities with the same lemmatized version
            if is_proper_name:
                entity_lemma = ' '.join(words_lemma)
                entity_raw = ' '.join(words_raw)
                if not (entity_lemma in title_groups):
                    title_groups[entity_lemma] = []

                # Check if the Wikipedia limit isn't overflowed
                if len(title_groups[entity_lemma]) < self.wikipedia_max_count:
                    title_groups[entity_lemma].append({
                        'entity_id': entity_id,
                        'text': entity_raw
                    })

        for entity_lemma in title_groups:
            # Form title from the lemmatized version and raw text
            titles = [entity_lemma]
            for item in title_groups[entity_lemma]:
                titles.append(item['text'])
            title = "|".join(titles)

            # Form params and send query
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'redirects',
                'titles': title
            }
            response = requests.post(self.url_api, params)
            if response.status_code == 200:

                # Read JSON response
                data = response.json()
                if "query" in data and "pages" in data["query"]:
                    pages = data["query"]["pages"]
                    for key in pages:
                        page_id = int(key)
                        if page_id > 0:
                            for item in title_groups[entity_lemma]:
                                entity_links[item['entity_id']] = pages[key]
                            break

        return entity_links

