import os


def from_tokens_to_colnll(tokens, document_id, offset=0):
    header = "#begin document ({0}); part 000{1}".format(document_id, os.linesep)
    is_entity_group = False
    lines = []
    entity_counter = offset
    for idx, token in enumerate(tokens):
        line = [document_id, '-']
        if token.IsEntity is not None:
            if not is_entity_group:
                    is_entity_group = True
                    if len(tokens) > idx + 1 and tokens[idx + 1].EntityID == token.EntityID:
                        line_format = '({0}'
                    else:
                        line_format = '({0})'
                        is_entity_group = False
                        entity_counter += 1
                    line[1] = line_format.format(entity_counter)
            else:
                if len(tokens) == idx + 1 or tokens[idx + 1].EntityID != token.EntityID:
                    line_format = '{0})'
                    line[1] = line_format.format(entity_counter)
                    entity_counter += 1
        lines.append(' '.join(line))

    pass