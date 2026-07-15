/* Hide the top navbar when scrolling down; reveal it when scrolling up. */
(function () {
    var lastScrollY = 0;
    var navbar = null;
    var navbarHeight = 0;

    function onScroll() {
        if (!navbar) {
            navbar = document.querySelector(".bd-header");
            if (navbar) {
                navbarHeight = navbar.offsetHeight;
            }
        }
        if (!navbar) return;

        var currentScrollY = window.scrollY;
        if (currentScrollY > lastScrollY && currentScrollY > navbarHeight) {
            // Scrolling down — hide the navbar
            navbar.classList.add("navbar-hidden");
        } else {
            // Scrolling up — show the navbar
            navbar.classList.remove("navbar-hidden");
        }
        lastScrollY = currentScrollY;
    }

    window.addEventListener("scroll", onScroll, { passive: true });
})();
