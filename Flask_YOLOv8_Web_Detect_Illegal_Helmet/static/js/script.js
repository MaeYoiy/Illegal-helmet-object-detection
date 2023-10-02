let slideIndex = 1;
showSlides(slideIndex);

function plusSlides(n) {
    showSlides(slideIndex += n);
}

function currentSlides(n) {
    showSlides(slideIndex = n);
}

function showSlides(n) {
    let i;
    let slides = document.getElementsByClassName("mySlides");
    let dots = document.getElementsByClassName("demo");
    let captionText = document.getElementById("caption");

    //ถ้า index เกิน 6 จะเริ่มนับ 1 ใหม่
    if (n > slides.length) {slideIndex = 1}
    //ถ้า index < 1 จะไปที่ index = 6 ทันที
    if (n < 1) {slideIndex = slides.length}

    //no display every slides
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }

    //loop ทุกตัวเพื่อแทนที่ class active
    for (i = 0; i < dots.length; i++) {
        //แทนที่ class active ด้วย string เปล่า ("")
        dots[i].className = dots[i].className.replace(" active", "");

    }

    //show slide at you choosed
    slides[slideIndex-1].style.display = "block";
    //ดึง text จาก demo และแสดงในส่วน caption-container
    captionText.innerHTML = dots[slideIndex - 1].alt;

    //เรากดตรงไหนก็ให้มันมี active ด้วย
    dots[slideIndex-1].className += " active";


}

