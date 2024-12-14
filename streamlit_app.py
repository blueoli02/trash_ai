#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1flUH367PZUDxW-k8d0Q6m6-yVpiRzNqM'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/FB5PDFQ/image.png",
            "https://i.ibb.co/dk3kMFy/image.jpg",
            "https://i.ibb.co/9b1mcWK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=aAXUEDrtj0U",
            "https://www.youtube.com/watch?v=T3l2aF0z6Bo",
            "https://www.youtube.com/watch?v=P2TADAfnV7w"
        ],
        'texts': [
            "골판지",
            "1.이물질 방지:물이나 기타 이물질이 묻지 않도록 주의!\n2.부착물 제거:택배 송장, 테이프, 철핀 등은 제거!\n3.부피 최소화:공간 절약을 위해 접어서 배출!\n4.올바른 분리:종이팩과는 다르게 분리하여 배출",
            "종이류, 세척이 불가능한 경우 일반쓰레기로 배출"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/DDNb1J4/image.webp",
            "https://i.ibb.co/dk3kMFy/image.jpg",
            "https://i.ibb.co/9b1mcWK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=jh157BWqlqs",
            "https://www.youtube.com/watch?v=T3l2aF0z6Bo",
            "https://www.youtube.com/watch?v=P2TADAfnV7w"
        ],
        'texts': [
            "금속",
            "금속캔과 고철류는 따로 분리해주세요.\n금속캔은 다른 재질 부분을 제거하고 분리배출함에 배출!\n고철류는 반드시 따로 비닐봉지에 배출!",
            "주의: 고무, 플라스틱이 부착되었거나 페인트통,폐유통 등 유해물질이 묻어있는 통은 재활용이 불가함!"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/q1vx6tg/image.png",
            "https://i.ibb.co/dk3kMFy/image.jpg",
            "https://i.ibb.co/9b1mcWK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=8Y8V-Yd-Cto",
            "https://www.youtube.com/watch?v=yYQCHZbrgB4",
            "https://www.youtube.com/watch?v=e_GoTN2q2M8"
        ],
        'texts': [
            "기타 쓰레기",
            "종량제봉투 배출! 그러나 확인은 필수!",
            "당신의 잠깐의 불편함이 지속적인 인류 안녕을 만듭니다."
        ]
    },
    labels[3]: {
        'images': [
            "https://i.ibb.co/Mp7phs2/image.jpg",
            "https://i.ibb.co/dk3kMFy/image.jpg",
            "https://i.ibb.co/9b1mcWK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=U7XSXf8bHfQ",
            "https://www.youtube.com/watch?v=T3l2aF0z6Bo",
            "https://www.youtube.com/watch?v=P2TADAfnV7w"
        ],
        'texts': [
            "유리",
            "일반 유리제품은 신문지에 싸서 재활용품으로 배출!\n내열유리는 종량제봉투나 특수규격마대를 구매하여 배출!",
            "깨진 유리는 재활용X :일반 종량제봉투가 찢어지지 않도록 신문지에 싸서 배출!"
        ]
    },
    labels[4]: {
        'images': [
            "https://i.ibb.co/px4P45n/image.webp",
            "https://i.ibb.co/dk3kMFy/image.jpg",
            "https://i.ibb.co/9b1mcWK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=U7XSXf8bHfQ",
            "https://www.youtube.com/watch?v=T3l2aF0z6Bo",
            "https://www.youtube.com/watch?v=P2TADAfnV7w"
        ],
        'texts': [
            "종이",
            "스프링 등 종이류와 다른 재질은 제거!",
            "영수증,전표,코팅지,오염된종이,기타 벽지 및 부직포 등은 종량제 봉투로 배출!"
        ]
    },
    labels[5]: {
        'images': [
            "https://i.ibb.co/4sctj5w/image.jpg",
            "https://i.ibb.co/dk3kMFy/image.jpg",
            "https://i.ibb.co/9b1mcWK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=9m4gnPozJVM",
            "https://www.youtube.com/watch?v=T3l2aF0z6Bo",
            "https://www.youtube.com/watch?v=P2TADAfnV7w"
        ],
        'texts': [
            "플라스틱",
            "무색페트병은 내용물, 라벨 제거 후 찌그러트려 뚜껑을 닫아 배출!\n플라스틱은 물로 헹구고 라벨,부속품 등 다른 재질은 분리한 후 투명,유색을 분리하여 배출!",
            "용기의 펌프(금속 스프링),칫솔,볼펜 등 다른 재질이 혼합,부착된 것은 종량제봉투로 GO GO"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

