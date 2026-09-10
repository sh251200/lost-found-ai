const lostForm = document.getElementById("lostForm");

if (lostForm) {

    lostForm.addEventListener("submit", async function(event) {

        // 페이지 새로고침 방지
        event.preventDefault();


        // ==========================================
        // 입력값 가져오기
        // ==========================================

        const itemName =
            document.getElementById("itemName").value;

        const location =
            document.getElementById("location").value;

        const date =
            document.getElementById("date").value;

        const description =
            document.getElementById("description").value;


        // ==========================================
        // 선택한 사진 가져오기
        // ==========================================

        const imageInput =
            document.getElementById("image");

        const image =
            imageInput.files[0];


        // 사진을 선택하지 않았는지 확인
        if (!image) {

            alert("분실물 사진을 선택해주세요.");

            return;
        }


        // ==========================================
        // 서버로 보낼 데이터 만들기
        // ==========================================

        const formData = new FormData();


        formData.append(
            "item_name",
            itemName
        );

        formData.append(
            "location",
            location
        );

        formData.append(
            "lost_date",
            date
        );

        formData.append(
            "description",
            description
        );

        formData.append(
            "image",
            image
        );


        // ==========================================
        // FastAPI 서버로 전송
        // ==========================================

        try {

           const response = await fetch(
            "https://lost-found-ai-0z75.onrender.com/lost-items",
                {
                    method: "POST",
                    body: formData
                }
            );


            // 서버 응답
            const result =
                await response.json();


            // ==========================================
            // 등록 성공
            // ==========================================

            if (response.ok) {

                alert(
                    "분실물이 정상적으로 등록되었습니다!\n\n" +
                    "등록번호: " +
                    result.id
                );


                // 입력창 초기화
                lostForm.reset();


                console.log(
                    "등록된 사진:",
                    result.image_path
                );

            }


            // ==========================================
            // 등록 실패
            // ==========================================

            else {

                alert(
                    "등록에 실패했습니다.\n\n" +
                    result.detail
                );

            }


        }


        // ==========================================
        // 서버 연결 실패
        // ==========================================

        catch (error) {

            console.error(error);

           alert(
                "서버에 연결할 수 없습니다.\n\n" +
                "잠시 후 다시 시도해주세요."

            );

        }

    });

}


// ==========================================
// 메인 화면 → 분실물 등록
// ==========================================

function goLost() {

    window.location.href = "lost.html";

}


// ==========================================
// 습득물 등록
// ==========================================

function goFound() {
    window.location.href = "found.html";
}

function startMatching() {
    window.location.href = "match.html";
}
