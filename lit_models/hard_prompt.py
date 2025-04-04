"""
Templates module for relation extraction tasks.
This module provides relation templates for various datasets in a unified format.
"""

import json
import os

# Dataset types
SEMEVAL = "semeval"
TACRED = "tacred"
TACREV = "tacrev"

# Template data maps
_templates = {
    # SemEval templates
    SEMEVAL: [
        {"token": ["whole is comprised of components."], "h": {"name": "whole"}, "t": {"name": "components"}, "relation": "Component-Whole(e2,e1)"},
        {"token": ["A and B are not related", "."], "h": {"name": "A"}, "t": {"name": "B"}, "relation": "Other"},
        {"token": ["agency using the instrument."], "h": {"name": "agency"}, "t": {"name": "instrument"}, "relation": "Instrument-Agency(e2,e1)"},
        {"token": ["member is in the collection."], "h": {"name": "member"}, "t": {"name": "collection"}, "relation": "Member-Collection(e1,e2)"},
        {"token": ["effect is caused by cause."], "h": {"name": "effect"}, "t": {"name": "cause"}, "relation": "Cause-Effect(e2,e1)"},
        {"token": ["the target of entity is destination ."], "h": {"name": "entity"}, "t": {"name": "destination"}, "relation": "Entity-Destination(e1,e2)"},
        {"token": ["content is in container."], "h": {"name": "content"}, "t": {"name": "container"}, "relation": "Content-Container(e1,e2)"},
        {"token": ["message is about the topic."], "h": {"name": "message"}, "t": {"name": "topic"}, "relation": "Message-Topic(e1,e2)"},
        {"token": ["producer make out a product."], "h": {"name": "producer"}, "t": {"name": "product"}, "relation": "Product-Producer(e2,e1)"},
        {"token": ["collection is a set of members."], "h": {"name": "collection"}, "t": {"name": "members"}, "relation": "Member-Collection(e2,e1)"},
        {"token": ["entity derived from the origin."], "h": {"name": "entity"}, "t": {"name": "origin"}, "relation": "Entity-Origin(e1,e2)"},
        {"token": ["cause that causes to effect."], "h": {"name": "cause"}, "t": {"name": "effect"}, "relation": "Cause-Effect(e1,e2)"},
        {"token": ["component is in the whole."], "h": {"name": "component"}, "t": {"name": "whole"}, "relation": "Component-Whole(e1,e2)"},
        {"token": ["topic is described through message."], "h": {"name": "topic"}, "t": {"name": "message"}, "relation": "Message-Topic(e2,e1)"},
        {"token": ["product is made by producer."], "h": {"name": "product"}, "t": {"name": "producer"}, "relation": "Product-Producer(e1,e2)"},
        {"token": ["origin is the source of entity."], "h": {"name": "origin"}, "t": {"name": "entity"}, "relation": "Entity-Origin(e2,e1)"},
        {"token": ["container is containing the content."], "h": {"name": "container"}, "t": {"name": "content"}, "relation": "Content-Container(e2,e1)"},
        {"token": ["instrument is used by agency."], "h": {"name": "instrument"}, "t": {"name": "agency"}, "relation": "Instrument-Agency(e1,e2)"},
        {"token": ["destination is the target of entity."], "h": {"name": "destination"}, "t": {"name": "entity"}, "relation": "Entity-Destination(e2,e1)"}
    ],
    
    # TACRED templates
    TACRED: [
        {"token": ["subject have title object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:title"},
        {"token": ["subject is the employee of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:employee_of"},
        {"token": ["subject have not relationship with object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "NA"},
        {"token": ["subject lives in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:countries_of_residence"},
        {"token": ["subject has the high level member object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:top_members/employees"},
        {"token": ["subject has a headquarter in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:country_of_headquarters"},
        {"token": ["subject has the religion object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:religion"},
        {"token": ["subject died because of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:cause_of_death"},
        {"token": ["subject is also known as object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:alternate_names"},
        {"token": ["subject was born in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:city_of_birth"},
        {"token": ["subject lives in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:cities_of_residence"},
        {"token": ["subject has a headquarter in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:city_of_headquarters"},
        {"token": ["subject has the age object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:age"},
        {"token": ["subject died in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:city_of_death"},
        {"token": ["subject has the nationality object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:origin"},
        {"token": ["subject is other family of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:other_family"},
        {"token": ["subject owns object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:subsidiaries"},
        {"token": ["subjectis the parent of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:children"},
        {"token": ["subject dissolved in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:dissolved"},
        {"token": ["subject lives in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:stateorprovinces_of_residence"},
        {"token": ["subject is the siblings of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:siblings"},
        {"token": ["subject is the spouse of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:spouse"},
        {"token": ["subject die in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:stateorprovince_of_death"},
        {"token": ["subject has the alternate name object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:alternate_names"},
        {"token": ["subject is a member of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:member_of"},
        {"token": ["subject has the parent company object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:parents"},
        {"token": ["subject has the website object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:website"},
        {"token": ["subject has the parent object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:parents"},
        {"token": ["subject was founded in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:founded"},
        {"token": ["subject has a headquarter in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:stateorprovince_of_headquarters"},
        {"token": ["subject studied in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:schools_attended"},
        {"token": ["subject has the member object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:members"},
        {"token": ["subject has political affiliation with object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:political/religious_affiliation"},
        {"token": ["subject has birthday on object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:date_of_birth"},
        {"token": ["subject was founded by object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:founded_by"},
        {"token": ["subject has shares hold in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:shareholders"},
        {"token": ["subject has the number of employees object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:number_of_employees/members"},
        {"token": ["subject was born in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:country_of_birth"},
        {"token": ["subject was born in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:stateorprovince_of_birth"},
        {"token": ["subject is convicted of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:charges"},
        {"token": ["subject died in the date object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:date_of_death"},
        {"token": ["subject died in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:country_of_death"}
    ],
    
    # TACREV templates - same as TACRED but with "no_relation" instead of "NA"
    TACREV: [
        {"token": ["subject have title object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:title"},
        {"token": ["subject is the employee of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:employee_of"},
        {"token": ["subject have not relationship with object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "no_relation"},
        {"token": ["subject lives in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:countries_of_residence"},
        {"token": ["subject has the high level member object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:top_members/employees"},
        {"token": ["subject has a headquarter in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:country_of_headquarters"},
        {"token": ["subject has the religion object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:religion"},
        {"token": ["subject died because of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:cause_of_death"},
        {"token": ["subject is also known as object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:alternate_names"},
        {"token": ["subject was born in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:city_of_birth"},
        {"token": ["subject lives in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:cities_of_residence"},
        {"token": ["subject has a headquarter in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:city_of_headquarters"},
        {"token": ["subject has the age object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:age"},
        {"token": ["subject died in the city object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:city_of_death"},
        {"token": ["subject has the nationality object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:origin"},
        {"token": ["subject is other family of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:other_family"},
        {"token": ["subject owns object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:subsidiaries"},
        {"token": ["subjectis the parent of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:children"},
        {"token": ["subject dissolved in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:dissolved"},
        {"token": ["subject lives in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:stateorprovinces_of_residence"},
        {"token": ["subject is the siblings of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:siblings"},
        {"token": ["subject is the spouse of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:spouse"},
        {"token": ["subject die in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:stateorprovince_of_death"},
        {"token": ["subject has the alternate name object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:alternate_names"},
        {"token": ["subject is a member of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:member_of"},
        {"token": ["subject has the parent company object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:parents"},
        {"token": ["subject has the website object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:website"},
        {"token": ["subject has the parent object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:parents"},
        {"token": ["subject was founded in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:founded"},
        {"token": ["subject has a headquarter in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:stateorprovince_of_headquarters"},
        {"token": ["subject studied in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:schools_attended"},
        {"token": ["subject has the member object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:members"},
        {"token": ["subject has political affiliation with object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:political/religious_affiliation"},
        {"token": ["subject has birthday on object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:date_of_birth"},
        {"token": ["subject was founded by object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:founded_by"},
        {"token": ["subject has shares hold in object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:shareholders"},
        {"token": ["subject has the number of employees object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "org:number_of_employees/members"},
        {"token": ["subject was born in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:country_of_birth"},
        {"token": ["subject was born in the state or province object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:stateorprovince_of_birth"},
        {"token": ["subject is convicted of object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:charges"},
        {"token": ["subject died in the date object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:date_of_death"},
        {"token": ["subject died in the country object."], "h": {"name": "subject"}, "t": {"name": "object"}, "relation": "per:country_of_death"}
    ]
}

def get_templates(dataset_type):
    """
    Get templates for a specific dataset type.
    
    Args:
        dataset_type: One of 'semeval', 'tacred', or 'tacrev'
    
    Returns:
        List of template data for the specified dataset
    """
    if dataset_type not in _templates:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be one of: {', '.join(_templates.keys())}")
    return _templates[dataset_type]

def load_templates(dataset_type):
    """
    Load templates and return them in JSON format (for backward compatibility).
    This mimics reading from files but uses the built-in templates.
    
    Args:
        dataset_type: One of 'semeval', 'tacred', or 'tacrev'
    
    Returns:
        List of template data as JSON strings
    """
    templates = get_templates(dataset_type)
    return [json.dumps(template) for template in templates]

def get_template_path(dataset_type):
    """
    For backward compatibility - returns a virtual path to template.
    
    Args:
        dataset_type: One of 'semeval', 'tacred', or 'tacrev'
    
    Returns:
        A string representing the "path" to the template
    """
    return f"virtual://templates/{dataset_type}"

def write_templates_to_string(dataset_type):
    """
    Write templates to a string, one JSON object per line.
    This emulates the content of the original template files.
    
    Args:
        dataset_type: One of 'semeval', 'tacred', or 'tacrev'
    
    Returns:
        String with template data, one JSON object per line
    """
    templates = get_templates(dataset_type)
    return "\n".join(json.dumps(template) for template in templates) 