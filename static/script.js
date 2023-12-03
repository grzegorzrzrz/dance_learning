window.transitionToPage = function(href) {
    document.querySelector('body').style.opacity = 0
    setTimeout(function() {
        window.location.href = href

    }, 500)

}

window.addEventListener('pageshow', function(event) {
    document.querySelector('body').style.opacity = 1
});