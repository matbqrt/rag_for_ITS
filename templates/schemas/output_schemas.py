from langchain.output_parsers import ResponseSchema

schemas = {
        'adaptative': [
                ResponseSchema(name="item",
                   description="""
                    This field contains the item's name if it is given or the item's type if a list of item is asked.
                    This field answers the question 'What ?'
                """),
                ResponseSchema(name="community",
                        description=""""
                        The community is a collection of items. It can either be a domain (collection of services) or a service (collection of objects).
                        This field answers the question 'Where ?'
                        This field is optional, return Null if no community is found.
                """),
        ],

        'filter': [
                ResponseSchema(name="type", description="""OPTIONNAL, IF NOT MENTIONNED EXPLICITLY RETURN NONE.
                                There are ONLY 4 differents types : 'service', 'object', 'domain' and 'diagram'
                                """),
                ResponseSchema(name="service name", description="""The service the object belongs to (optionnal, if not mentionned return None).
                                                                It allows the user to give indication on which service he wants you to extract data.
                                                                !!! SERVICE AND DOMAIN CAN'T HAVE THE SAME NAME !!!
                                """),
                ResponseSchema(name='object name', description="""OPTIONNAL, IF NOT MENTIONNED EXPLICITLY RETURN NONE.
                                !!! ONLY IF TYPE IS OBJECT AND NO SERVICE NAME GIVEN !!!. SERVICE AND OBJECT CANNOT HAVE THE SAME NAME !!!
                                RARELY ASKED.
                                It is the object name."""),
                ResponseSchema(name="domain name", description="""OPTIONNAL, IF NOT MENTIONNED EXPLICITLY RETURN NONE.
                                The area the service belongs to.
                                !!! SERVICE AND DOMAIN CAN'T HAVE THE SAME NAME !!!
                                """),
                ResponseSchema(name="class name", description="""OPTIONNAL, IF NOT MENTIONNED EXPLICITLY RETURN NONE.
                                Collection of physical objects. There is five possible class : field, support, center, personal and vehicle
                                !!! ONLY IF NATURE IS PHYSICAL AND TYPE IS OBJECT !!!
                                """),
                ResponseSchema(name="nature", description="""OPTIONNAL, IF NOT MENTIONNED EXPLICITLY RETURN NONE.
                                ONLY if type is object.
                                There are ONLY 3 different natures of object : 'physical', 'functional' and 'information flow' .                                           
                                """),
        ],

        'history': [
            ResponseSchema(name="relevant docs",
                   description="""
                    Past retrieved documents might be useful to answer the question.
                    Given the list of retrieved documents, store in 'relevant docs' the name of the documents
                    that might be useful to answer the new question.
                    If no document seems relevant, output an empty list.
                """)
                ]
        
        

}