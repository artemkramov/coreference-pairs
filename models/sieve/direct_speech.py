import re
import uuid


# Class to work with direct speech
class DirectSpeech:

    # Set of regular expression to find direct speech
    regular_expression = []

    def __init__(self):
        # Init list of templates to form regular expressions
        # for different cases
        regex = []

        # "П",-а.
        # "П!",-а.
        # "П?",-а.
        # "П...",-а.
        regex.append(r"\{left_quote}(?P<direct1>[^\{right_quote}]{{0,}})\{right_quote},\{dash}(?P<author1>[^\.]{{0,}})\.")

        # А: "П".
        # А: "П!".
        # А: "П?".
        # А: "П…".
        regex.append(r"(?P<author1>[^\.]{{0,}})\:\{left_quote}(?P<direct1>[^\.]{{0,}})\{right_quote}\.")

        # "П, — а, — п".
        # "П, — а. — П".
        # "П… (?,!) — а. — П… (?,!)"
        # "П… (?,!) — а, а: — П… (?,!)"
        regex.append(
            r"\{left_quote}(?P<direct1>[^\.\{dash}]{{0,}})[\?\,\!]\{dash}(?P<author1>[^\.]{{0,}})[\.\,\:]\{dash}(?P<direct2>[^\.]{{0,}})\{right_quote}\.")

        # А: "П", - а.
        # А: "П!" - а.
        # А: "П?" - а.
        # А: "П…" - а.
        regex .append(r"(?P<author1>[^\.\{left_quote}]{{0,}})\:\{left_quote}(?P<direct1>.*?)\{right_quote}\,\{dash}(?P<author2>[^\.]{{0,}})\.")
        for r in regex:
            self.regular_expression.extend(self.form_regex_with_different_combinations(r))

    # Form regular expressions from different combinations of parameters
    @staticmethod
    def form_regex_with_different_combinations(regex):
        combinations = []

        # Set different variants of quotes and dashes
        symbol_variants = [[['"', '"'], ['«', '»']], [['‒'], ['–'], ['—'], ['-']]]

        # Loop through both parts
        for first_part in symbol_variants[0]:
            for second_part in symbol_variants[1]:

                # Copy first parameter and extend it with the second part
                params = first_part[:]
                params.extend(second_part)

                # Format template to create regular expression
                combinations.append(regex.format(left_quote=params[0], right_quote=params[1], dash=params[2]))
        return combinations

    # Find direct speech groups from the tokens given
    def find_direct_speech_groups(self, tokens):
        response = {}

        # Dictionary which set pairs like startPosition-token
        token_positions = {}

        # Convert tokens to the text without spaces
        text = ''
        for idx, token in enumerate(tokens):
            text += token.RawText
            token_positions[len(text) - len(token.RawText)] = idx

        # List which contains all ranges that were matched from regular expressions
        match_ranges = []

        # Loop through each regular expression and apply it
        for regex in self.regular_expression:
            matches = re.finditer(regex, text, re.MULTILINE)
            for match in matches:
                is_new = True

                # Get groups as a dictionary
                groups = match.groupdict()

                # Loop through each group
                for key in groups:

                    # Get position of the group
                    start = match.start(key)
                    end = match.end(key)

                    # Check if the group doesn't intersect with any existing group
                    r = set(range(start, end))
                    for match_range in match_ranges:
                        if len(match_range.intersection(r)) > 0:
                            is_new = False
                            break
                # If match doesn't intersect with other existing groups
                if is_new:
                    group_id = str(uuid.uuid4())
                    for key in groups:
                        start = match.start(key)
                        end = match.end(key)
                        match_ranges.append(set(range(start, end)))

                        selected_tokens = tokens[token_positions[start]:token_positions[end]]
                        role = 'author'
                        if 'direct' in key:
                            role = 'direct'
                        for token in selected_tokens:
                            response[token.WordOrder] = {'group_id': group_id, 'role': role}

        return response
