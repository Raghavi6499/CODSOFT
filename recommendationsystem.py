
careers = {
    "Data Scientist": ["python", "statistics", "machine learning", "data", "problem solving"],
    "Software Engineer": ["coding", "programming", "python", "java", "algorithms"],
    "Graphic Designer": ["creativity", "design", "photoshop", "illustrator", "drawing"],
    "Digital Marketer": ["seo", "social media", "content", "marketing", "communication"],
    "Civil Engineer": ["construction", "mathematics", "physics", "design", "structural"],
    "Doctor": ["medicine", "biology", "anatomy", "patient care", "healthcare"],
    "Teacher": ["teaching", "communication", "leadership", "knowledge", "mentoring"],
    "Entrepreneur": ["business", "leadership", "innovation", "management", "strategy"]
}

def recommend_career(user_skills):
    scores = {}
    for career, skills in careers.items():
        match_count = len(set(user_skills) & set(skills))
        scores[career] = match_count

    sorted_careers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [career for career, score in sorted_careers if score > 0][:3]


print(" Career Path Recommendation System")
print("Enter your skills or interests (comma separated): ")
user_input = input(" ")


user_skills = [skill.strip().lower() for skill in user_input.split(",")]
recommendations = recommend_career(user_skills)

if recommendations:
    print("\n Based on your skills, you might enjoy these careers:")
    for i, career in enumerate(recommendations, start=1):
        print(f"{i}. {career}")
else:
    print("\nSorry, no matching career found. Try adding more skills.")

