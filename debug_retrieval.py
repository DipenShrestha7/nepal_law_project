from chatbot.test import detect_act_title_query, collect_act_title_matches

print(detect_act_title_query('What is the Financial Act of Nepal?'))
print(detect_act_title_query('आर्थिक ऐन'))
print(detect_act_title_query('Money Bill under Article 110'))
print(detect_act_title_query('धन सम्बन्धी विधेयक'))
